import argparse
import hashlib
import io
import shutil
import sys
import time
import zipfile
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .models import load_encoder
from .sae_campaign import digest
from .sae_encoders import captured_sites, fmcib_prefix, random_encoder
from .sae_geometry import fmcib_sampling_affine
from .sae_sites import SITES, material, translation, volume_with_geometry


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
CASE_KEYS = {
    "fmcib": ["context_trained", "context_random", "context_valid", "context_distance", "context_variables", "layer2_random", "layer2_distance", "layer2_variables"],
    "vista": ["stage2_random", "stage3_random", "stage2_distance", "stage3_distance", "stage2_variables", "stage3_variables"],
}
VARIABLES = ["hu_mean", "hu_std", "gradient_mean", "air_fraction", "dense_fraction", "context_hu_mean"]


def configuration():
    return read_json(CONFIG_PATH)


def root_directory():
    storage, _, _ = context()
    return Path(storage["runs"]) / configuration()["campaign"]


def members():
    storage, _, _ = context()
    table = pd.read_csv(Path(storage["activations"]) / configuration()["previous_suite"] / "fmcib/full_members.csv", dtype={"PatientID": str})
    assert set(table.split) == {"train", "development"}
    assert set(table.query("split == 'train'").PatientID).isdisjoint(table.query("split == 'development'").PatientID)
    return table


def stable_value(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def viewer_members(table):
    storage, _, _ = context()
    development = pd.read_csv(Path(storage["runs"]) / "20261009_feature_exploration/members.csv", dtype={"PatientID": str})
    chosen = table.loc[table.identifier.isin(development.identifier)].assign(viewer_split="confirmation")
    assert len(chosen) == 128 and set(chosen.split) == {"development"}
    train = table.query("split == 'train'").assign(order=lambda frame: frame.identifier.map(lambda value: stable_value(f"viewer-record:{value}")))
    first = train.sort_values("order").groupby("PatientID", sort=False).head(1)
    first = first.assign(order=first.PatientID.map(lambda value: stable_value(f"viewer-patient:{value}")))
    counts = development.label.value_counts().to_dict()
    picked = pd.concat([first.query("label == @label").sort_values("order").head(count) for label, count in sorted(counts.items())])
    assert len(picked) == 128 and picked.PatientID.is_unique
    picked = picked.drop(columns="order").assign(viewer_split="discovery")
    return pd.concat([picked.sort_values("identifier"), chosen.sort_values("identifier")]).reset_index(drop=True)


def target_component(mask):
    labels, count = ndimage.label(mask > 0, structure=np.ones((3, 3, 3)))
    assert count > 0, "Mask has no foreground"
    center = tuple(np.asarray(mask.shape) // 2)
    chosen = labels[center]
    if chosen == 0:
        points = np.argwhere(labels > 0)
        chosen = labels[tuple(points[np.argmin(np.square(points - np.asarray(center)).sum(1))])]
    return labels == chosen, count


def sample_mask(native_mask, native_affine, input_affine, shape):
    coordinates = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    native = nib.affines.apply_affine(np.linalg.inv(native_affine) @ input_affine, coordinates)
    values = ndimage.map_coordinates(native_mask.astype(np.float32), native.T, order=0, mode="constant", cval=0.0)
    return values.reshape(shape) > 0.5


def signed_distance(mask, spacing):
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return np.where(mask, -inside, outside).astype(np.float32)


def grid_samples(volume, stride, first, count):
    index = stride * (first + np.arange(count))
    return volume[np.ix_(index, index, index)]


def image_variables(hu, spacing, stride, first, count):
    mean = ndimage.uniform_filter(hu, stride, mode="nearest")
    square = ndimage.uniform_filter(hu * hu, stride, mode="nearest")
    gradient = np.sqrt(sum(np.square(axis) for axis in np.gradient(hu, *spacing)))
    values = [
        mean,
        np.sqrt(np.maximum(square - mean * mean, 0)),
        ndimage.uniform_filter(gradient, stride, mode="nearest"),
        ndimage.uniform_filter((hu < -900).astype(np.float32), stride, mode="nearest"),
        ndimage.uniform_filter((hu > 100).astype(np.float32), stride, mode="nearest"),
        ndimage.uniform_filter(hu, 3 * stride, mode="nearest"),
    ]
    return np.stack([grid_samples(value, stride, first, count) for value in values]).astype(np.float32)


def nodule_references(image, component, affine):
    spacing = nib.affines.voxel_sizes(affine)
    voxel_volume = float(np.prod(spacing))
    volume = float(component.sum() * voxel_volume)
    points = np.argwhere(component).astype(np.float64)
    corners = (points[:, None, :] + np.asarray([[a, b, c] for a in (-.5, .5) for b in (-.5, .5) for c in (-.5, .5)])[None]).reshape(-1, 3) * spacing
    hull = corners[ConvexHull(corners).vertices]
    maximum_diameter = float(np.sqrt(np.max(np.square(hull[:, None, :] - hull[None, :, :]).sum(-1))))
    padded = np.pad(component, 1).astype(np.float32)
    vertices, faces, _, _ = measure.marching_cubes(padded, 0.5, spacing=tuple(spacing))
    surface = float(measure.mesh_surface_area(vertices, faces))
    values = image[component]
    distance = ndimage.distance_transform_edt(~component, sampling=spacing)
    near, far = (distance > 0) & (distance <= 5), (distance > 5) & (distance <= 10)
    return {
        "volume_mm3": volume, "equivalent_diameter_mm": float((6 * volume / np.pi) ** (1 / 3)),
        "maximum_diameter_mm": maximum_diameter, "surface_area_mm2": surface,
        "sphericity": float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface),
        "hu_mean": float(values.mean()), "hu_std": float(values.std()),
        "hu_p10": float(np.percentile(values, 10)), "hu_p90": float(np.percentile(values, 90)),
        "solid_fraction": float((values > -300).mean()),
        "ring_0_5mm_hu_mean": float(image[near].mean()) if near.any() else None,
        "ring_5_10mm_hu_mean": float(image[far].mean()) if far.any() else None,
        "touches_crop_border": bool(component[[0, -1]].any() or component[:, [0, -1]].any() or component[:, :, [0, -1]].any()),
    }


def canonical(values, affine):
    orientation = nib.orientations.io_orientation(affine)
    transform = nib.orientations.ornt_transform(orientation, nib.orientations.axcodes2ornt(("R", "A", "S")))
    return nib.orientations.apply_orientation(values, transform), affine @ nib.orientations.inv_ornt_aff(transform, values.shape)


def fmcib_inputs(row, config):
    if str(ROOT / "third_party/fmcib_source") not in sys.path:
        sys.path.insert(0, str(ROOT / "third_party/fmcib_source"))
    from fmcib.preprocessing import get_transforms

    settings = material("fmcib")["preprocessing"]["settings"]
    source = nib.load(row.image)
    center_ras = nib.affines.apply_affine(source.affine, np.asarray(source.shape) // 2)
    transform = get_transforms(spatial_size=tuple(settings["shape"]))
    values = {"image_path": str(row.image), "coordX": -float(center_ras[0]), "coordY": -float(center_ras[1]), "coordZ": float(center_ras[2])}
    oriented = Compose(transform.transforms[:5])(values)
    image = oriented["image_path"]
    original_affine = np.asarray(image.affine).copy()
    array = image.as_tensor()[0].numpy()
    center = nib.affines.apply_affine(np.linalg.inv(original_affine), center_ras).astype(int)
    deployed = Compose(transform.transforms[5:])(oriented).to(dtype=torch.float32).numpy()[0]
    half = np.asarray(settings["shape"])[::-1] // 2
    start, end = np.maximum(center - half, 0), np.minimum(center + half, array.shape)
    padding = np.maximum(np.asarray(settings["shape"]) - (end - start)[::-1], 0) // 2
    permutation = np.eye(4)[:, [2, 1, 0, 3]]
    deployed_affine = fmcib_sampling_affine(source.affine, source.shape, original_affine @ translation(start) @ permutation @ translation(-padding))
    clamped = bool(np.any(center - half < 0) or np.any(center + half > np.asarray(array.shape)))
    cube_size, offset = config["context"]["cube_size"], config["context"]["cube_offset"]
    cube_start = center - offset
    cube = np.zeros((cube_size,) * 3, dtype=np.float32)
    low, high = np.maximum(cube_start, 0), np.minimum(cube_start + cube_size, array.shape)
    cube[tuple(slice(a, b) for a, b in zip(low - cube_start, high - cube_start))] = array[tuple(slice(a, b) for a, b in zip(low, high))]
    cube = np.ascontiguousarray(cube.transpose(2, 1, 0))
    cube_affine = fmcib_sampling_affine(source.affine, source.shape, original_affine @ translation(cube_start) @ permutation)
    first, count, field = config["context"]["first_position"], config["context"]["positions"], config["context"]["receptive_field"]
    centers = 4 * (first + np.arange(count))
    valid_axes = []
    for axis in range(3):
        source_axis = 2 - axis
        valid_axes.append((cube_start[source_axis] + centers - field // 2 >= 0) & (cube_start[source_axis] + centers + field // 2 <= array.shape[source_axis] - 1))
    valid = valid_axes[0][:, None, None] & valid_axes[1][None, :, None] & valid_axes[2][None, None, :]
    return {"cube": cube, "cube_affine": cube_affine, "deployed": deployed, "deployed_affine": deployed_affine, "valid": valid, "clamped": clamped, "center": center.tolist(), "cube_start": cube_start.tolist(), "oriented_shape": list(array.shape), "source": source}


def prepare_record(row, config, viewer):
    fmcib = fmcib_inputs(row, config)
    source = fmcib.pop("source")
    image = np.asarray(source.dataobj, dtype=np.float32)
    mask_image = nib.load(row.mask)
    assert np.allclose(mask_image.affine, source.affine, atol=1e-4) and mask_image.shape == source.shape
    component, components = target_component(np.asarray(mask_image.dataobj))
    references = nodule_references(image, component, source.affine)
    first, count = config["context"]["first_position"], config["context"]["positions"]
    cube_spacing = nib.affines.voxel_sizes(fmcib["cube_affine"])
    cube_mask = sample_mask(component, source.affine, fmcib["cube_affine"], fmcib["cube"].shape)
    cube_hu = fmcib["cube"] * 3072 - 1024
    deployed_spacing = nib.affines.voxel_sizes(fmcib["deployed_affine"])
    deployed_mask = sample_mask(component, source.affine, fmcib["deployed_affine"], fmcib["deployed"].shape)
    deployed_hu = fmcib["deployed"] * 3072 - 1024
    vista_tensor, vista_geometry = volume_with_geometry(row.image, "vista")
    vista_affine = np.asarray(vista_geometry["input_affine_ras"])
    vista_values = vista_tensor.numpy()[0]
    low, high = material("vista")["preprocessing"]["settings"]["intensity"]
    vista_hu = vista_values * (high - low) + low
    vista_spacing = nib.affines.voxel_sizes(vista_affine)
    vista_mask = sample_mask(component, source.affine, vista_affine, vista_values.shape)
    cube_distance = signed_distance(cube_mask, cube_spacing)
    deployed_distance = signed_distance(deployed_mask, deployed_spacing)
    vista_distance = signed_distance(vista_mask, vista_spacing)
    arrays = {
        "fmcib": {
            "context_valid": fmcib["valid"],
            "context_distance": grid_samples(cube_distance, 4, first, count),
            "context_variables": image_variables(cube_hu, cube_spacing, 4, first, count),
            "layer2_distance": grid_samples(deployed_distance, 8, 0, 7),
            "layer2_variables": image_variables(deployed_hu, deployed_spacing, 8, 0, 7),
        },
        "vista": {
            "stage2_distance": grid_samples(vista_distance, 4, 0, 12),
            "stage3_distance": grid_samples(vista_distance, 8, 0, 6),
            "stage2_variables": image_variables(vista_hu, vista_spacing, 4, 0, 12),
            "stage3_variables": image_variables(vista_hu, vista_spacing, 8, 0, 6),
        },
    }
    context_affine = fmcib["cube_affine"] @ np.diag([4, 4, 4, 1]) @ translation([first] * 3)
    geometry = {
        "identifier": row.identifier, "PatientID": row.PatientID, "split": row.split, "label": int(row.label),
        "fmcib_center_index": fmcib["center"], "cube_start": fmcib["cube_start"], "oriented_shape": fmcib["oriented_shape"],
        "deployed_clamped": fmcib["clamped"], "context_valid_positions": int(fmcib["valid"].sum()),
        "cube_affine_ras": fmcib["cube_affine"].tolist(), "context_grid_affine_ras": context_affine.tolist(),
        "fmcib_deployed_affine_ras": fmcib["deployed_affine"].tolist(), "layer2_grid_affine_ras": (fmcib["deployed_affine"] @ np.diag([8, 8, 8, 1])).tolist(),
        "vista_input_affine_ras": vista_affine.tolist(), "stage2_grid_affine_ras": (vista_affine @ np.diag([4, 4, 4, 1])).tolist(), "stage3_grid_affine_ras": (vista_affine @ np.diag([8, 8, 8, 1])).tolist(),
        "mask_components": int(components), "nodule_voxels_context_cube": int(cube_mask.sum()), "nodule_voxels_vista_input": int(vista_mask.sum()),
        "references": references,
    }
    views = None
    if viewer:
        views = {}
        for scope, hu, mask, affine in [("fmcib_context", cube_hu, cube_mask, fmcib["cube_affine"]), ("fmcib", deployed_hu, deployed_mask, fmcib["deployed_affine"]), ("vista", vista_hu, vista_mask, vista_affine)]:
            values, canonical_affine = canonical(hu, affine)
            labels, _ = canonical(mask.astype(np.uint8), affine)
            views[scope] = {"input_hu": values.astype(np.float32), "mask": labels.astype(np.uint8), "input_affine_ras": affine, "canonical_affine_ras": canonical_affine}
    return {"identifier": row.identifier, "cube": fmcib["cube"], "deployed": fmcib["deployed"], "vista": vista_values.astype(np.float32), "arrays": arrays, "geometry": geometry, "views": views}


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, frame, config, viewer_identifiers):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.viewer = set(viewer_identifiers)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        torch.set_num_threads(1)
        row = next(self.frame.iloc[[index]].itertuples())
        return prepare_record(row, self.config, row.identifier in self.viewer)


def unchanged(value):
    return value


def vista_sites(model, tensor):
    return captured_sites(model, "vista", tensor, {site: SITES["vista"][site]["module"] for site in ["stage2", "stage3"]})


def complete_case(path, keys):
    if not path.exists() or not path.with_suffix(".json").exists():
        return False
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist()) == {f"{key}.npy" for key in keys}


def save_npz(path, values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    temporary.replace(path)


def extract(limit=None, workers=6):
    config = configuration()
    root = root_directory()
    storage, _, _ = context()
    previous = Path(storage["activations"]) / config["previous_suite"]
    table = members()
    viewers = viewer_members(table)
    if limit:
        table = pd.concat([table.query("split == @split").groupby("label", sort=True).head(limit) for split in ["train", "development"]])
        table = pd.concat([table, viewers.head(2).drop(columns="viewer_split")]).drop_duplicates("identifier")
    for model in ["fmcib", "vista"]:
        (root / "cases" / model).mkdir(parents=True, exist_ok=True)
    viewers.to_csv(root / "viewer_members.csv", index=False)
    source = {
        "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(Path(__file__)),
        "geometry_sha256": sha256(ROOT / "src/sclvmi/sae_geometry.py"), "sites_sha256": sha256(ROOT / "src/sclvmi/sae_sites.py"),
        "members_sha256": sha256(previous / "fmcib/full_members.csv"), "viewer_members": viewers.identifier.tolist(),
        "fmcib_material": material("fmcib")["preprocessing"], "vista_material": material("vista")["preprocessing"],
        "random_encoder": config["random_encoder"], "test_used": False,
    }
    ensure = root / "cases/source.json"
    if ensure.exists() and read_json(ensure) != source:
        assert not any((root / "cases/fmcib").glob("*.npz")), "Changed extraction source requires a new campaign"
        ensure.rename(ensure.with_name(f"source_superseded_{time.time_ns()}.json"))
    if not ensure.exists():
        write_json(ensure, source)
    pending = table.loc[[not (complete_case(root / "cases/fmcib" / f"{value}.npz", CASE_KEYS["fmcib"]) and complete_case(root / "cases/vista" / f"{value}.npz", CASE_KEYS["vista"])) for value in table.identifier]]
    print("PENDING", len(pending), "OF", len(table), flush=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trained_fmcib = load_encoder("fmcib")[0].cuda()
    random_fmcib = random_encoder("fmcib", config["random_encoder"]["initialization_seed"]).cuda()
    random_vista = random_encoder("vista", config["random_encoder"]["initialization_seed"]).cuda()
    loader = torch.utils.data.DataLoader(RecordDataset(pending, config, viewers.identifier), batch_size=None, collate_fn=unchanged, num_workers=workers, persistent_workers=False, prefetch_factor=4 if workers else None)
    started = time.monotonic()
    check_every = 1 if limit else 50
    verified = []
    for index, item in enumerate(tqdm(loader, total=len(pending), desc="interpretation inputs", mininterval=5)):
        identifier = item["identifier"]
        geometry = item["geometry"]
        with torch.inference_mode():
            cube = torch.from_numpy(item["cube"])[None, None].cuda()
            deployed = torch.from_numpy(item["deployed"])[None, None].cuda()
            vista = torch.from_numpy(item["vista"])[None, None].cuda()
            context_trained = fmcib_prefix(trained_fmcib, cube, "layer1")[0, :, 5:17, 5:17, 5:17]
            context_random = fmcib_prefix(random_fmcib, cube, "layer1")[0, :, 5:17, 5:17, 5:17]
            layer2_random = fmcib_prefix(random_fmcib, deployed, "layer2")[0]
            vista_random = vista_sites(random_vista, vista)
            if not geometry["deployed_clamped"] and index % check_every == 0:
                deployed_layer1 = fmcib_prefix(trained_fmcib, deployed, "layer1")[0]
                with np.load(io.BytesIO((previous / "fmcib/cases" / f"{identifier}.npz").read_bytes())) as saved:
                    reference = saved["layer1"]
                deployed_error = float(np.abs(deployed_layer1.cpu().numpy() - reference).max())
                correspondence = float((deployed_layer1[:, 5:9, 5:9, 5:9] - context_trained[:, 4:8, 4:8, 4:8]).abs().max())
                assert deployed_error < 1e-4 and correspondence < 1e-4, (identifier, deployed_error, correspondence)
                geometry["deployed_check"] = {"deployed_layer1_max_error": deployed_error, "context_interior_max_error": correspondence}
                verified.append({"identifier": identifier, **geometry["deployed_check"]})
        fmcib_values = {"context_trained": context_trained.cpu().numpy(), "context_random": context_random.cpu().numpy(), "layer2_random": layer2_random.cpu().numpy(), **item["arrays"]["fmcib"]}
        vista_values = {"stage2_random": vista_random["stage2"][0].cpu().numpy(), "stage3_random": vista_random["stage3"][0].cpu().numpy(), **item["arrays"]["vista"]}
        for key, value in [*fmcib_values.items(), *vista_values.items()]:
            assert not np.isnan(value).any() if value.dtype != bool else True, key
            assert key.endswith("distance") or value.dtype == bool or np.isfinite(value).all(), key
        assert fmcib_values["context_trained"].shape == (512, 12, 12, 12) and fmcib_values["layer2_random"].shape == (1024, 7, 7, 7)
        assert vista_values["stage2_random"].shape == (192, 12, 12, 12) and vista_values["stage3_random"].shape == (384, 6, 6, 6)
        for model, values in [("fmcib", fmcib_values), ("vista", vista_values)]:
            path = root / "cases" / model / f"{identifier}.npz"
            save_npz(path, values)
            write_json(path.with_suffix(".json"), geometry)
        if item["views"] is not None:
            for scope, view in item["views"].items():
                path = root / "viewer_inputs" / scope / f"{identifier}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.h5")
                with h5py.File(temporary, "w") as saved:
                    saved.create_dataset("input_hu", data=view["input_hu"], compression="gzip", compression_opts=1)
                    saved.create_dataset("mask", data=view["mask"], compression="gzip", compression_opts=1)
                    saved.attrs["input_affine_ras"] = view["input_affine_ras"]
                    saved.attrs["canonical_affine_ras"] = view["canonical_affine_ras"]
                    saved.attrs["identifier"] = identifier
                temporary.replace(path)
        if index % 20 == 0 or index + 1 == len(pending):
            write_json(root / "progress/extraction.json", {"task": "extraction", "completed": len(table) - len(pending) + index + 1, "total": len(table), "new_records": index + 1, "records_per_second": (index + 1) / (time.monotonic() - started), "updated_at": timestamp()})
    if verified:
        frame = pd.DataFrame(verified)
        path = root / ("smoke_deployed_checks.csv" if limit else "deployed_checks.csv")
        frame = pd.concat([pd.read_csv(path), frame]).drop_duplicates("identifier") if path.exists() else frame
        frame.to_csv(path, index=False)
    references = []
    for row in table.itertuples():
        geometry = read_json(root / "cases/fmcib" / f"{row.identifier}.json")
        references.append({"identifier": row.identifier, "PatientID": row.PatientID, "split": row.split, "label": row.label, "deployed_clamped": geometry["deployed_clamped"], "context_valid_positions": geometry["context_valid_positions"], "mask_components": geometry["mask_components"], **geometry["references"]})
    pd.DataFrame(references).to_csv(root / ("smoke_references.csv" if limit else "nodule_references.csv"), index=False)
    table.to_csv(root / ("smoke_members.csv" if limit else "members.csv"), index=False)
    write_json(root / ("smoke_extraction.json" if limit else "extraction.json"), {"state": "completed", "records": len(table), "new_records": len(pending), "seconds": time.monotonic() - started, "source_sha256": sha256(ensure), "completed_at": timestamp(), "test_used": False})


def sampler(seed, model, site, patient):
    return np.random.default_rng(stable_value(f"{seed}:{model}:{site}:{patient}"))


def write_matrix(destination, model, site, encoder, channels, member_path, offsets_by_split, extra):
    for split in ["train", "development"]:
        np.save(destination / f"{split}_offsets.npy", offsets_by_split[split])
    train = np.load(destination / "train.npy", mmap_mode="r")
    train_offsets = offsets_by_split["train"]
    mean = np.zeros(channels, dtype=np.float64)
    square = np.zeros_like(mean)
    for start, end in zip(train_offsets[:-1], train_offsets[1:]):
        values = np.asarray(train[start:end], dtype=np.float64)
        assert np.isfinite(values).all()
        mean += values.mean(0)
        square += np.square(values).mean(0)
    mean /= len(train_offsets) - 1
    square /= len(train_offsets) - 1
    variance = np.maximum(square - mean ** 2, 0)
    scale = float(np.sqrt(variance.mean()))
    assert scale > 0
    np.savez(destination / "normalization.npz", mean=mean.astype(np.float32), scale=np.float32(scale), variance=variance)
    members_frame = pd.read_csv(member_path, dtype={"PatientID": str})
    metadata = {"model": model, "site": site, "encoder": encoder, "channels": channels, "kind": "spatial", "members_sha256": sha256(member_path), "sampling": configuration()["sampling"], "test_used": False, "normalization": "equal training-patient channel mean and global RMS", "splits": {}, "files": {}, **extra}
    for split in ["train", "development"]:
        offsets = offsets_by_split[split]
        metadata["splits"][split] = {"patients": len(offsets) - 1, "tokens": int(offsets[-1]), "cases": int((members_frame.split == split).sum()), "unique_positions": int(offsets[-1])}
    for path in sorted(destination.glob("*.np*")):
        metadata["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    metadata["identity"] = digest(metadata)
    metadata["created_at"] = timestamp()
    write_json(destination / "matrix.json", metadata)
    print("MATRIX_READY", model, site, encoder, metadata["splits"], flush=True)


def deployed_matrix(root, model, site, positions, verify_patients=3):
    config = configuration()
    storage, _, _ = context()
    previous = Path(storage["activations"]) / config["previous_suite"] / model / "cases"
    trained = root / "matrices" / model / site
    destination = root / "matrices_random" / model / site
    if (destination / "matrix.json").exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    member_path = root / "members.csv"
    table = pd.read_csv(member_path, dtype={"PatientID": str})
    channels = config["models"][model][site]["channels"]
    offsets_by_split = {}
    for split in ["train", "development"]:
        selected = table.query("split == @split").sort_values(["PatientID", "identifier"])
        groups = list(selected.groupby("PatientID", sort=True))
        budget = config["sampling"][f"{split}_tokens_per_patient"]
        lengths = [min(budget, len(frame) * positions) for _, frame in groups]
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        np.testing.assert_array_equal(offsets, np.load(trained / f"{split}_offsets.npy"))
        offsets_by_split[split] = offsets
        output = np.lib.format.open_memmap(destination / f"{split}.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), channels))
        reference = np.load(trained / f"{split}.npy", mmap_mode="r")
        for patient_index, (patient, rows) in enumerate(tqdm(groups, desc=f"{model} {site} {split} random matrix", mininterval=5)):
            chosen = sampler(config["sampling"]["seed"], model, site, patient).choice(len(rows) * positions, size=int(offsets[patient_index + 1] - offsets[patient_index]), replace=False)
            for case_index, row in enumerate(rows.itertuples()):
                selected_indices = np.flatnonzero(chosen // positions == case_index)
                if len(selected_indices) == 0:
                    continue
                with np.load(root / "cases" / model / f"{row.identifier}.npz") as saved:
                    tokens = saved[f"{site}_random"].reshape(channels, -1).T
                output[offsets[patient_index] + selected_indices] = tokens[chosen[selected_indices] % positions]
                if patient_index < verify_patients:
                    with np.load(previous / f"{row.identifier}.npz") as saved:
                        trained_tokens = saved[site].reshape(channels, -1).T
                    np.testing.assert_array_equal(reference[offsets[patient_index] + selected_indices], trained_tokens[chosen[selected_indices] % positions])
        output.flush()
        del output
    write_matrix(destination, model, site, "random", channels, member_path, offsets_by_split, {"paired_with": read_json(trained / "matrix.json")["identity"], "sampling_verified_patients": verify_patients})


def context_matrices(root):
    config = configuration()
    member_path = root / "members.csv"
    table = pd.read_csv(member_path, dtype={"PatientID": str})
    channels = config["models"]["fmcib"]["layer1_context"]["channels"]
    destinations = {encoder: root / directory / "fmcib/layer1_context" for encoder, directory in [("trained", "matrices"), ("random", "matrices_random")]}
    if all((path / "matrix.json").exists() for path in destinations.values()):
        return
    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)
    offsets_by_split = {}
    for split in ["train", "development"]:
        selected = table.query("split == @split").sort_values(["PatientID", "identifier"])
        groups = list(selected.groupby("PatientID", sort=True))
        budget = config["sampling"][f"{split}_tokens_per_patient"]
        valid = {row.identifier: read_json(root / "cases/fmcib" / f"{row.identifier}.json")["context_valid_positions"] for row in selected.itertuples()}
        lengths = [min(budget, sum(valid[value] for value in frame.identifier)) for _, frame in groups]
        assert min(lengths) > 0
        offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        offsets_by_split[split] = offsets
        outputs = {encoder: np.lib.format.open_memmap(path / f"{split}.npy", mode="w+", dtype=np.float32, shape=(int(offsets[-1]), channels)) for encoder, path in destinations.items()}
        token_case, token_position, token_distance = np.zeros(int(offsets[-1]), dtype=np.int32), np.zeros(int(offsets[-1]), dtype=np.int32), np.zeros(int(offsets[-1]), dtype=np.float32)
        identifiers = selected.identifier.tolist()
        lookup = {value: index for index, value in enumerate(identifiers)}
        for patient_index, (patient, rows) in enumerate(tqdm(groups, desc=f"context {split} matrices", mininterval=5)):
            loaded = []
            for row in rows.itertuples():
                with np.load(io.BytesIO((root / "cases/fmcib" / f"{row.identifier}.npz").read_bytes())) as saved:
                    flat_valid = np.flatnonzero(saved["context_valid"].reshape(-1))
                    loaded.append((row.identifier, flat_valid, saved["context_trained"].reshape(channels, -1).T, saved["context_random"].reshape(channels, -1).T, saved["context_distance"].reshape(-1)))
            total = sum(len(item[1]) for item in loaded)
            chosen = sampler(config["sampling"]["seed"], "fmcib", "layer1_context", patient).choice(total, size=int(offsets[patient_index + 1] - offsets[patient_index]), replace=False)
            boundaries = np.cumsum([0] + [len(item[1]) for item in loaded])
            for case_index, (identifier, flat_valid, trained_tokens, random_tokens, distance) in enumerate(loaded):
                selected_indices = np.flatnonzero((chosen >= boundaries[case_index]) & (chosen < boundaries[case_index + 1]))
                if len(selected_indices) == 0:
                    continue
                positions = flat_valid[chosen[selected_indices] - boundaries[case_index]]
                rows_out = offsets[patient_index] + selected_indices
                outputs["trained"][rows_out] = trained_tokens[positions]
                outputs["random"][rows_out] = random_tokens[positions]
                token_case[rows_out] = lookup[identifier]
                token_position[rows_out] = positions
                token_distance[rows_out] = distance[positions]
        for output in outputs.values():
            output.flush()
        del outputs
        for path in destinations.values():
            np.savez(path / f"{split}_tokens.npz", identifiers=np.asarray(identifiers), case=token_case, position=token_position, distance=token_distance)
    for encoder, path in destinations.items():
        write_matrix(path, "fmcib", "layer1_context", encoder, channels, member_path, offsets_by_split, {"native_shape": [12, 12, 12], "positions": "complete receptive-field positions only"})


def copy_deployed(root):
    config = configuration()
    storage, _, _ = context()
    source_root = Path(storage["runs"]) / config["deployed_matrix_campaign"] / "matrices"
    for model, site in [("fmcib", "layer2"), ("vista", "stage2"), ("vista", "stage3")]:
        destination = root / "matrices" / model / site
        if (destination / "matrix.json").exists():
            continue
        source = source_root / model / site
        metadata = read_json(source / "matrix.json")
        destination.mkdir(parents=True, exist_ok=True)
        for name, record in metadata["files"].items():
            shutil.copyfile(source / name, destination / name)
            assert sha256(destination / name) == record["sha256"], name
        shutil.copyfile(source / "matrix.json", destination / "matrix.json")
        print("COPIED", model, site, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["extract", "matrices", "copy"], required=True)
    parser.add_argument("--smoke", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = root_directory()
    if args.stage == "extract":
        extract(args.smoke, args.workers)
    elif args.stage == "copy":
        copy_deployed(root)
    else:
        context_matrices(root)
        deployed_matrix(root, "fmcib", "layer2", 343)
        deployed_matrix(root, "vista", "stage2", 1728)
        deployed_matrix(root, "vista", "stage3", 216)


if __name__ == "__main__":
    main()

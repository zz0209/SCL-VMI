import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .flexict_sae import nodule_surface, window_token_centres
from .nodule_context import ball
from .nodule_mask import target_component


CONFIG = ROOT / "configs/flexict_text.json"
PATCH, GRID, WINDOW, SPACING = 8, 20, 160, 2.0
TOKENS = GRID ** 3
BLOCK = 16
FLIP = np.diag([-1.0, -1.0, 1.0])


def configuration():
    return read_json(CONFIG)


def run_root():
    return Path(context()[0]["runs"]) / configuration()["run_id"]


def input_directory(dataset, smoke=False):
    storage = context()[0]
    if dataset == "luna25":
        return Path(storage["preprocessed"]) / configuration()["inputs_suite"] / "inputs"
    return Path(storage["preprocessed"]) / "lidc" / configuration()["run_id"] / ("smoke" if smoke else "")


def output_directory(dataset, stage, smoke):
    return run_root() / ("smoke" if smoke else "") / dataset / stage


def input_batches(dataset, smoke):
    batches = sorted(input_directory(dataset, smoke).glob("batch_*.h5"))
    return batches[:1] if smoke else batches


def concept_rows(spec):
    rows = []
    for group, entries in spec["vocabulary"].items():
        if group == "control_rule":
            continue
        rows += [{"group": group, "concept": concept, "phrase": phrase} for concept, phrasings in entries.items() for phrase in phrasings]
    return pd.DataFrame(rows)


def load_vocabulary():
    with np.load(run_root() / "vocabulary.npz") as saved:
        return {key: saved[key] for key in saved.files}


# Local: text directions.

@torch.inference_mode()
def vocabulary():
    from .vlm_text_check import load_model

    spec = configuration()
    output = run_root()
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, identity = load_model()
    table = concept_rows(spec)
    assert table.phrase.is_unique
    template = spec["text"]["template"]

    def encode(prompts):
        return torch.cat([model.encode_text(prompts[start:start + 16]).float() for start in range(0, len(prompts), 16)]).cpu().numpy()

    present = encode([template["present"].format(phrase) for phrase in table.phrase])
    absent = encode([template["absent"].format(phrase) for phrase in table.phrase])
    single = np.stack([model.encode_text([template["present"].format(phrase)]).float().cpu().numpy()[0] for phrase in table.phrase[:16]])
    concepts = list(dict.fromkeys(table.concept))
    direction = np.stack([(present - absent)[table.concept.to_numpy() == concept].mean(0) for concept in concepts]).astype(np.float32)
    groups = [table.group[table.concept == concept].iloc[0] for concept in concepts]
    np.savez(output / "vocabulary.npz", concepts=np.array(concepts), groups=np.array(groups), phrases=table.phrase.to_numpy(dtype=str), phrase_concepts=table.concept.to_numpy(dtype=str),
             present=present, absent=absent, direction=direction)
    cosine = present @ absent.T
    write_json(output / "vocabulary.json", {**identity, "concepts": len(concepts), "phrases": len(table), "groups": pd.Series(groups).value_counts().to_dict(),
                                           "batch_max_abs_difference": float(np.abs(single - present[:16]).max()), "present_absent_cosine_median": float(np.median(np.diag(cosine))),
                                           "direction_norm_range": [float(np.linalg.norm(direction, axis=1).min()), float(np.linalg.norm(direction, axis=1).max())],
                                           "sha256": sha256(output / "vocabulary.npz"), "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp()})
    print(f"vocabulary: {len(concepts)} concepts, {len(table)} phrasings, batch difference {np.abs(single - present[:16]).max():.2e}", flush=True)


# Window geometry: window voxel index (z, y, x) to RAS millimetres, the mapping of flexict_sae.window_token_centres.

def window_affine(size, spacing, origin, direction, start):
    shape = np.array([int(round(count * step / SPACING)) for count, step in zip(size, spacing)])[::-1]
    pad = (np.maximum(shape, WINDOW) - shape) // 2
    matrix = np.asarray(direction, dtype=float).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = FLIP @ matrix @ (SPACING * np.eye(3)[::-1])
    affine[:3, 3] = FLIP @ (np.asarray(origin, dtype=float) + matrix @ (SPACING * (np.asarray(start) - pad)[::-1]))
    return affine, pad


def token_index():
    grid = np.arange(GRID) * PATCH + (PATCH - 1) / 2
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    return np.stack([z, y, x], -1).reshape(-1, 3)


def metal_points(array, threshold):
    # array in (z, y, x) order; voxels at or above the threshold whose in-plane 3x3 median is too (isolated noise spikes in bone fail it); returned as (x, y, z) indices
    candidates = np.argwhere(array >= threshold)
    if not len(candidates):
        return candidates
    offsets = np.stack(np.meshgrid([0], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3)
    neighbours = np.clip(candidates[:, None, :] + offsets[None], 0, np.asarray(array.shape) - 1)
    return candidates[np.median(array[tuple(neighbours.transpose(2, 0, 1))], axis=1) >= threshold][:, ::-1]


def metal_tokens(points, spacing, pad, start, voxel_volume):
    # native voxel index (x, y, z) to the official 2.0 mm grid: same origin and direction, so the resampled index is index * spacing / 2.0
    window = np.rint(points * np.asarray(spacing) / SPACING)[:, ::-1].astype(int) + pad - np.asarray(start)
    inside = np.all((window >= 0) & (window < WINDOW), 1)
    token = np.ravel_multi_index(tuple((window[inside] // PATCH).T), (GRID,) * 3)
    return np.bincount(token, minlength=TOKENS).astype(np.float32) * voxel_volume


def axial_fill(mask, largest_voxels=None):
    filled = np.stack([ndimage.binary_fill_holes(plane) for plane in mask])
    if largest_voxels is None:
        return filled
    in_plane = np.zeros((3, 3, 3), dtype=bool)
    in_plane[1] = ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(filled & ~mask, structure=in_plane)
    sizes = np.bincount(labels.ravel())
    sizes[0] = largest_voxels + 1
    return mask | (sizes[labels] <= largest_voxels)


def window_masks(hu):
    # lung and body rules of configs/flexict_text.json geometry
    in_plane = np.zeros((3, 3, 3), dtype=bool)
    in_plane[1] = ndimage.generate_binary_structure(2, 1)
    labels, _ = ndimage.label(hu < -500, structure=in_plane)
    border = np.unique(np.concatenate([labels[:, [0, -1], :].ravel(), labels[:, :, [0, -1]].ravel()]))
    labels, _ = ndimage.label((labels > 0) & ~np.isin(labels, border))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    lung = np.isin(labels, np.flatnonzero(sizes >= 0.2 * sizes.max())) if sizes.max() > 0 else np.zeros(hu.shape, dtype=bool)
    lung = axial_fill(lung, int(1000 / SPACING ** 2))
    labels, count = ndimage.label(ndimage.binary_opening(hu > -500, structure=ball(2)), structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body = labels == np.argmax(sizes) if count else np.zeros(hu.shape, dtype=bool)
    return lung, axial_fill(body | lung)


def blocks(values):
    return values.reshape(GRID, PATCH, GRID, PATCH, GRID, PATCH).transpose(0, 2, 4, 1, 3, 5).reshape(TOKENS, -1)


def signed_distances(centres, masks):
    values = []
    for path in masks:
        image, component, tree = nodule_surface(path)
        gap, _ = tree.query(centres)
        voxel = np.rint(np.linalg.solve(image.affine, np.c_[centres, np.ones(len(centres))].T).T[:, :3]).astype(int)
        within = np.all((voxel >= 0) & (voxel < np.asarray(component.shape)), 1)
        inside = np.zeros(len(centres), dtype=bool)
        inside[within] = component[tuple(voxel[within].T)]
        values.append(np.where(inside, -gap, gap))
    return values


def window_geometry(hu, affine, metal, nodule_masks, member_mask, centre_token, settings):
    lung, body = window_masks(hu)
    lung_tokens, body_tokens = blocks(lung), blocks(body)
    values = blocks(hu.astype(np.float32))
    lung_count = lung_tokens.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        laa = ((values < -950) & lung_tokens).sum(1) / lung_count
        haa = ((values >= -600) & (values <= -250) & lung_tokens).sum(1) / lung_count
    centres = token_index() @ affine[:3, :3].T + affine[:3, 3]
    lung_volume = float(lung.sum() * SPACING ** 3 / 1e6)
    failed = lung_volume < settings["minimum_lung_litres"]
    soft = body & ~lung & (hu > -150) & (hu < 150)
    difference = np.diff(hu.astype(np.float32), axis=2)[soft[:, :, 1:] & soft[:, :, :-1]]
    noise = float(1.4826 * np.median(np.abs(difference - np.median(difference))) / np.sqrt(2))
    result = {"lung_fraction": lung_tokens.mean(1).astype(np.float32), "body_fraction": body_tokens.mean(1).astype(np.float32), "laa": laa.astype(np.float32), "haa": haa.astype(np.float32),
              "metal": metal.astype(np.float32), "centres": centres.astype(np.float32)}
    scalars = {"lung_litres": lung_volume, "lung_failed": failed, "metal_mm3": float(metal.sum()), "noise_hu": noise}
    if lung.any() and not failed:
        index = np.argwhere(lung)
        physical = index @ affine[:3, :3].T + affine[:3, 3]
        low, high = physical[:, 2].min(), physical[:, 2].max()
        middle = (physical[:, 0].min() + physical[:, 0].max()) / 2
        lung_values = hu[lung]
        scalars.update({"pd15": float(np.percentile(lung_values, 15)), "haa_fraction": float(((lung_values >= -600) & (lung_values <= -250)).mean()), "lung_low_mm": float(low), "lung_high_mm": float(high)})
        result["height"] = ((centres[:, 2] - low) / (high - low)).astype(np.float32)
        result["medial"] = (-np.abs(centres[:, 0] - middle)).astype(np.float32)
        result["band"] = (result["lung_fraction"] == 0) & (result["body_fraction"] >= 0.5) & (centres[:, 2] >= low) & (centres[:, 2] <= high)
    else:
        scalars.update({"pd15": np.nan, "haa_fraction": np.nan, "lung_low_mm": np.nan, "lung_high_mm": np.nan})
        result["height"] = np.full(TOKENS, np.nan, dtype=np.float32)
        result["medial"] = np.full(TOKENS, np.nan, dtype=np.float32)
        result["band"] = np.zeros(TOKENS, dtype=bool)
    distances = signed_distances(centres, nodule_masks)
    result["distance_any"] = (np.min(distances, 0) if distances else np.full(TOKENS, np.inf)).astype(np.float32)
    result["distance_member"] = (signed_distances(centres, [member_mask])[0] if member_mask else np.full(TOKENS, np.nan)).astype(np.float32)
    result["centre_token"] = np.int32(centre_token)
    metal_index = np.flatnonzero(metal >= settings["metal_token_mm3"])
    if len(metal_index):
        gap, _ = cKDTree(centres[metal_index]).query(centres)
        result["metal_reference"] = (result["body_fraction"] >= 0.5) & (metal == 0) & (gap > 32.0)
    else:
        result["metal_reference"] = np.zeros(TOKENS, dtype=bool)
    return result, scalars


def geometry_settings():
    spec = configuration()["geometry"]
    return {"minimum_lung_litres": 1.5, "metal_hu": spec["metal_hu"], "metal_token_mm3": 8.0}


# Local: LIDC official whole-CT windows.

def lidc_scan(member, view, lidc_root, settings, metal_hu):
    import pylidc as pl

    from .flexict_luna25 import cut_window, official_volume, window_placement
    from .sae_lidc import scan_volume

    scan = pl.query(pl.Scan).filter(pl.Scan.id == int(member["scan_id"])).first()
    volume, affine, reasons = scan_volume(scan, lidc_root, settings)
    assert not reasons, (member["identifier"], reasons)
    image = sitk.GetImageFromArray(np.ascontiguousarray(volume.transpose(2, 1, 0)))
    spacing = tuple(float(value) for value in np.diag(affine)[:3])
    image.SetSpacing(spacing)
    image.SetOrigin(tuple(float(value) for value in affine[:3, 3]))
    resampled, clamped, mean, std = official_volume(image, view["spacing_mm"])
    centre = (float(member["center_x_mm"]), float(member["center_y_mm"]), float(member["center_z_mm"]))
    nodule = np.asarray(resampled.TransformPhysicalPointToContinuousIndex(centre))[::-1]
    official, start, pad, local, readout = window_placement(np.asarray(clamped.shape), nodule, view)
    points = metal_points(volume.transpose(2, 1, 0), metal_hu)
    keys = list(dict.fromkeys([tuple(official.tolist()), tuple(start.tolist())]))
    windows = [{"window": cut_window(clamped, np.asarray(key), pad, np.asarray(view["window"]), int(clamped.min())), "start": np.asarray(key), "official": key == tuple(official.tolist()),
                "metal": metal_tokens(points, spacing, pad, key, float(np.prod(spacing)))} for key in keys]
    record = {"window": keys.index(tuple(start.tolist())), "official_window": keys.index(tuple(official.tolist())), "readout": readout, "nodule": local.astype(np.float32)}
    header = {"size": np.asarray(image.GetSize()), "spacing": np.asarray(spacing), "origin": np.asarray(image.GetOrigin()), "direction": np.asarray(image.GetDirection())}
    return member["identifier"], f"lidc_{member['scan_id']}", windows, record, mean, std, header


def lidc_prepare(workers, smoke):
    from .flexict_luna25 import specification

    spec = configuration()
    storage = context()[0]
    lidc_config = read_json(ROOT / "configs/sae_lidc.json")
    source = Path(storage["preprocessed"]) / "lidc" / spec["lidc_run"]
    members = pd.read_csv(source / "members.csv", dtype={"PatientID": str})
    assert members.scan_id.is_unique and len(members) == 582
    members = members.sort_values("identifier").reset_index(drop=True)
    if smoke:
        members = members.head(4)
    view = specification()["views"]["w20"]
    output = input_directory("lidc", smoke)
    output.mkdir(parents=True, exist_ok=True)
    size = 64
    chunks = [members.iloc[start:start + size] for start in range(0, len(members), size)]
    started = time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        for number, chunk in enumerate(chunks):
            path = output / f"batch_{number:03d}.h5"
            if path.exists():
                continue
            results = list(pool.map(lidc_scan, chunk.to_dict("records"), [view] * len(chunk), [Path(storage["lidc_root"])] * len(chunk), [lidc_config["nodules"]] * len(chunk),
                                    [spec["geometry"]["metal_hu"]] * len(chunk)))
            temporary = path.with_suffix(".tmp")
            options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
            with h5py.File(temporary, "w") as handle:
                handle.attrs.update({"members_sha256": sha256(source / "members.csv"), "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp()})
                handle.create_dataset("identifier", data=np.array([item[0] for item in results], dtype="S"))
                group = handle.create_group("w20")
                rows = {key: [] for key in ["windows", "scan", "mean", "std", "start", "official", "metal", "size", "spacing", "origin", "direction"]}
                records = {key: [] for key in ["window_index", "official_index", "readout", "nodule"]}
                for identifier, scan, windows, record, mean, std, header in results:
                    offset = len(rows["scan"])
                    for entry in windows:
                        rows["windows"].append(entry["window"])
                        rows["scan"].append(scan)
                        rows["mean"].append(mean)
                        rows["std"].append(std)
                        rows["start"].append(entry["start"])
                        rows["official"].append(entry["official"])
                        rows["metal"].append(entry["metal"])
                        for key in ["size", "spacing", "origin", "direction"]:
                            rows[key].append(header[key])
                    records["window_index"].append(offset + record["window"])
                    records["official_index"].append(offset + record["official_window"])
                    records["readout"].append(record["readout"])
                    records["nodule"].append(record["nodule"])
                group.create_dataset("windows", data=np.stack(rows["windows"]), chunks=(1, *view["window"]), **options)
                group.create_dataset("scan", data=np.array(rows["scan"], dtype="S"))
                for key in ["mean", "std", "official"]:
                    group.create_dataset(key, data=np.array(rows[key]))
                for key in ["start", "metal", "size", "spacing", "origin", "direction"]:
                    group.create_dataset(key, data=np.stack(rows[key]))
                for key, value in records.items():
                    group.create_dataset(key, data=np.array(value) if key.endswith("index") else np.stack(value))
            temporary.replace(path)
            done = sum(len(item) for item in chunks[:number + 1])
            elapsed = time.monotonic() - started
            print(f"lidc prepare: {path.name} ({number + 1}/{len(chunks)}), {done}/{len(members)} scans, {elapsed / 60:.1f} min, eta {elapsed / done * (len(members) - done) / 60:.1f} min", flush=True)
    write_json(output / "inputs.json", {"run": spec["run_id"], "members": len(members), "batches": [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.glob("batch_*.h5"))],
                                        "source": spec["lidc_run"], "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp()})


# Local GPU: final-layer tokens, embeddings, concept token scores and production SAE codes.

def production_dictionaries():
    spec = configuration()
    root = Path(context()[0]["runs"]) / spec["sae_campaign"]
    production = read_json(root / "report" / "production.json")["sites"][spec["site"]]
    assert production["accepted"] and production["group"] == spec["sae"]["group"], production["group"]
    entries = sorted(production["dictionaries"], key=lambda entry: entry["seed"])
    assert entries[0]["seed"] == spec["sae"]["reference_seed"] and len(entries) == 8
    for entry in entries:
        assert sha256(root / entry["path"]) == entry["sha256"], entry["run_id"]
    return root, entries


@torch.inference_mode()
def encode(dataset, smoke):
    from .flexict_luna25 import encode as forward, readout, specification, window_input
    from .models import load_flexict, published
    from .sae import SpatialDictionary

    spec = configuration()
    output = output_directory(dataset, "encoded", smoke)
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = Path(published("flexict")["directory"]) / "ct_3d_vlm.pth"
    model, _ = load_flexict("ct_3d_vlm.pth")
    model.cuda()
    projection = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]["vlm_vision_projection.weight"].float().cuda()
    words = load_vocabulary()
    direction = torch.from_numpy(words["direction"]).cuda()
    token_matrix = projection[:, 864:].T @ direction.T
    class_matrix = projection[:, :864].T @ direction.T
    sae_root, entries = production_dictionaries()
    dictionaries = [SpatialDictionary(sae_root / entry["path"]) for entry in entries]
    features = dictionaries[0].weights["W_enc"].shape[1]
    concepts = direction.shape[0]
    identity = {"vocabulary_sha256": sha256(run_root() / "vocabulary.npz"), "dictionaries": json.dumps([[entry["run_id"], entry["sha256"]] for entry in entries]),
                "checkpoint_sha256": sha256(checkpoint), "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__))}
    checks, started, windows_done, vlm = [], time.monotonic(), 0, None
    view = specification()["views"]["w20"]
    batches = input_batches(dataset, smoke)
    for number, path in enumerate(batches):
        target = output / path.name
        if target.exists():
            continue
        with h5py.File(path, "r") as handle:
            group = handle["w20"]
            count = min(6, len(group["scan"])) if smoke else len(group["scan"])
            # the classifier reads readout_tokens^3 tokens per record from its readout window (official, or moved near the border)
            identifiers, positions = handle["identifier"][:].astype(str), group["readout"][:]
            readers = {}
            for record, window in enumerate(group["window_index"][:]):
                if window < count:
                    readers.setdefault(int(window), []).append(record)
            read_ids, read_windows, read_tokens = [], [], []
            temporary = target.with_suffix(".tmp")
            with h5py.File(temporary, "w") as out:
                out.attrs.update({**identity, "source_sha256": sha256(path), "created_at": timestamp()})
                out.create_dataset("scan", data=group["scan"][:count])
                out.create_dataset("official", data=group["official"][:count])
                out.create_dataset("start", data=group["start"][:count])
                created = {name: out.create_dataset(name, shape, dtype=dtype, chunks=chunk) for name, shape, dtype, chunk in [
                    ("cls", (count, 864), "f4", None), ("mean", (count, 864), "f4", None), ("embedding", (count, 1024), "f4", None), ("scan_score", (count, concepts), "f4", None),
                    ("local", (count, TOKENS, concepts), "f2", (1, TOKENS, concepts)), ("seed_sum", (count, 8, features), "f4", None), ("seed_square", (count, 8, features), "f4", None),
                    ("seed_count", (count, 8, features), "i2", None), ("seed_max", (count, 8, features), "f4", None),
                    ("seed_weighted", (count, 8, features, concepts), "f4", (1, 1, features, concepts))]}
                indices, values, offsets, base = [], [], [np.zeros(1, dtype=np.int64)], 0
                for window in range(count):
                    x = window_input(group["windows"][window], float(group["mean"][window]), float(group["std"][window]))
                    (patch, cls), = forward(model, x, [BLOCK])
                    patch, cls = patch.float(), cls.float()
                    for record in readers.get(window, []):
                        read_ids.append(identifiers[record])
                        read_windows.append(window)
                        read_tokens.append(readout(patch, np.array([GRID] * 3), view, positions[record]).cpu().numpy())
                    mean = patch.mean(0)
                    joint = torch.cat([cls, mean]) @ projection.T
                    embedding = F.normalize(joint, dim=-1)
                    local = patch @ token_matrix
                    created["cls"][window] = cls.cpu().numpy()
                    created["mean"][window] = mean.cpu().numpy()
                    created["embedding"][window] = embedding.cpu().numpy()
                    created["scan_score"][window] = (embedding @ direction.T).cpu().numpy()
                    created["local"][window] = local.half().cpu().numpy()
                    summary = {name: [] for name in ["seed_sum", "seed_square", "seed_count", "seed_max", "seed_weighted"]}
                    for seed, dictionary in enumerate(dictionaries):
                        codes = dictionary.encode(patch)
                        summary["seed_sum"].append(codes.sum(0))
                        summary["seed_square"].append(codes.square().sum(0))
                        summary["seed_count"].append((codes > 0).sum(0))
                        summary["seed_max"].append(codes.amax(0))
                        summary["seed_weighted"].append(codes.T @ local)
                        if seed == 0:
                            active = codes > 0
                            token, feature = torch.nonzero(active, as_tuple=True)
                            indices.append(feature.to(torch.int16).cpu().numpy())
                            values.append(codes[token, feature].cpu().numpy())
                            cumulative = torch.cumsum(active.sum(1), 0).cpu().numpy()
                            offsets.append(base + cumulative)
                            base += int(cumulative[-1])
                            reference = codes
                    for name, items in summary.items():
                        created[name][window] = torch.stack(items).cpu().numpy()
                    if len(checks) < 2:
                        if vlm is None:
                            from .vlm_text_check import load_model

                            vlm, _ = load_model()
                        checks.append(window_check(vlm, x, patch, cls, joint, local, class_matrix, direction, reference, indices[-1], values[-1], np.r_[0, cumulative], dictionaries[0]))
                        print("CHECK", checks[-1], flush=True)
                        if len(checks) == 2:
                            vlm = None
                            torch.cuda.empty_cache()
                    windows_done += 1
                    if (window + 1) % 25 == 0:
                        elapsed = time.monotonic() - started
                        print(f"encode {dataset}: {path.name} window {window + 1}/{count}, {windows_done} windows, {elapsed / windows_done:.2f} s/window", flush=True)
                out.create_dataset("codes/index", data=np.concatenate(indices))
                out.create_dataset("codes/value", data=np.concatenate(values))
                out.create_dataset("codes/offsets", data=np.concatenate(offsets))
                out.create_dataset("readout/identifier", data=np.array(read_ids, dtype="S"))
                out.create_dataset("readout/window", data=np.array(read_windows))
                out.create_dataset("readout/tokens", data=np.stack(read_tokens))
            temporary.replace(target)
        elapsed = time.monotonic() - started
        print(f"encode {dataset}: {path.name} ({number + 1}/{len(batches)}) {count} windows, {target.stat().st_size / 1e9:.2f} GB, {elapsed / 60:.1f} min", flush=True)
    if checks:
        write_json(output / f"checks_{timestamp()[:19].replace(':', '')}.json", {"checks": checks, "dataset": dataset, "created_at": timestamp()})


def window_check(vlm, x, patch, cls, joint, local, class_matrix, direction, codes, index, value, offsets, dictionary):
    official = vlm.encode_image(x.cuda()).float()[0]
    embedding = F.normalize(joint, dim=-1)
    unnormalised = joint @ direction.T
    rebuilt = local.mean(0) + cls @ class_matrix
    sparse = torch.zeros_like(codes)
    rows = torch.repeat_interleave(torch.arange(len(offsets) - 1, device=codes.device), torch.from_numpy(np.diff(offsets)).to(codes.device))
    sparse[rows, torch.from_numpy(index.astype(np.int64)).to(codes.device)] = torch.from_numpy(value).to(codes.device)
    result = {"embedding_max_abs_difference": float((official - embedding).abs().max()), "score_max_relative_difference": float(((rebuilt - unnormalised).abs() / unnormalised.abs().clamp_min(1e-6)).max()),
              "codes_equal": bool(torch.equal(sparse, codes)), "codes_match_encode": bool(torch.equal(dictionary.encode(patch), codes)), "active_per_token": float((codes > 0).sum(1).float().mean())}
    assert result["embedding_max_abs_difference"] < 1e-4 and result["score_max_relative_difference"] < 1e-3 and result["codes_equal"] and result["codes_match_encode"], result
    return result


# Local CPU: lung, body, metal and nodule geometry of every encoded window.

def luna_cases():
    storage, _, dataset = context()
    cases = pd.read_csv(input_directory("luna25") / "cases.csv", dtype={"PatientID": str})
    cases = cases[cases.split != "test"].copy()
    cases["mask"] = [str(dataset / "labelsTr" / f"{identifier}.nii.gz") for identifier in cases.identifier]
    cases["scan"] = [str(Path(storage["luna25_root"]) / "images" / "luna25_images" / f"{series}.mha") for series in cases.SeriesInstanceUID]
    return cases


def luna_members():
    spec = configuration()
    frame = pd.read_csv(Path(context()[0]["runs"]) / "20261014_sae_interpretation" / "profiles" / "fmcib_layer2" / "members.csv", dtype={"PatientID": str})
    assert frame.PatientID.is_unique and len(frame) == 1686, spec["members"]["luna25"]
    return frame


def lidc_tables():
    source = Path(context()[0]["preprocessed"]) / "lidc" / configuration()["lidc_run"]
    members = pd.read_csv(source / "members.csv", dtype={"PatientID": str})
    nodules = pd.read_csv(source / "nodules.csv", dtype={"PatientID": str})
    members["mask"] = [str(source / path) for path in members["mask"]]
    nodules["mask"] = [str(source / path) for path in nodules["mask"]]
    return members, nodules


def scan_geometry(task):
    settings = geometry_settings()
    path, windows, header, masks, member, metal_source = task
    if metal_source:
        image = sitk.ReadImage(metal_source)
        array = sitk.GetArrayFromImage(image)
        points = metal_points(array, settings["metal_hu"])
        voxel_volume = float(np.prod(image.GetSpacing()))
        assert np.allclose(image.GetSpacing(), header["spacing"]) and np.allclose(image.GetOrigin(), header["origin"])
    results = []
    with h5py.File(path, "r") as handle:
        group = handle["w20"]
        for window, centre_token, member_mask in windows:
            start = group["start"][window]
            affine, pad = window_affine(header["size"], header["spacing"], header["origin"], header["direction"], start)
            if metal_source:
                assert np.allclose(window_token_centres(image, start), token_index() @ affine[:3, :3].T + affine[:3, 3], atol=1e-3), path
            metal = metal_tokens(points, header["spacing"], pad, start, voxel_volume) if metal_source else group["metal"][window]
            hu = group["windows"][window]
            values, scalars = window_geometry(hu, affine, metal, masks, member_mask, centre_token, settings)
            results.append((window, values, scalars))
    volumes = {}
    for mask in member["volume_masks"]:
        image = sitk.ReadImage(mask)
        component, _ = target_component(sitk.GetArrayFromImage(image))
        volumes[Path(mask).name.split(".")[0]] = float(component.sum() * np.prod(image.GetSpacing()))
    return results, volumes


def geometry_tasks(dataset, path, smoke):
    count = 6 if smoke else None
    with h5py.File(path, "r") as handle:
        group = handle["w20"]
        scans = group["scan"][:count].astype(str)
        starts = group["start"][:count]
        official = group["official"][:count]
        identifiers = handle["identifier"][:].astype(str)
        window_index, official_index, nodule = group["window_index"][:], group["official_index"][:], group["nodule"][:]
        headers = {key: group[key][:count] for key in ["size", "spacing", "origin", "direction"]} if dataset == "lidc" else None
    # nodule centre in each window of its scan: readout-window position moved by the start difference
    centres = {}
    for record, identifier in enumerate(identifiers):
        if window_index[record] >= len(scans):
            continue
        for window in np.flatnonzero(scans == scans[window_index[record]]):
            position = nodule[record] + starts[window_index[record]] - starts[window]
            token = np.floor(position / PATCH).astype(int)
            centres[(identifier, window)] = int(np.ravel_multi_index(tuple(token), (GRID,) * 3)) if np.all((token >= 0) & (token < GRID)) else -1
    tasks = []
    if dataset == "luna25":
        cases, members = luna_cases(), luna_members()
        member_ids = set(members.identifier)
        for scan in dict.fromkeys(scans):
            rows = cases[cases.SeriesInstanceUID == scan]
            reader = sitk.ImageFileReader()
            reader.SetFileName(rows.scan.iloc[0])
            reader.ReadImageInformation()
            header = {"size": np.asarray(reader.GetSize()), "spacing": np.asarray(reader.GetSpacing()), "origin": np.asarray(reader.GetOrigin()), "direction": np.asarray(reader.GetDirection())}
            member = rows[rows.identifier.isin(member_ids)]
            assert len(member) <= 1
            items = []
            for window in np.flatnonzero(scans == scan):
                if len(member):
                    items.append((int(window), centres.get((member.identifier.iloc[0], window), -1), member["mask"].iloc[0]))
                else:
                    items.append((int(window), -1, None))
            tasks.append((str(path), items, header, rows["mask"].tolist(), {"volume_masks": rows["mask"].tolist()}, rows.scan.iloc[0]))
    else:
        members, nodules = lidc_tables()
        lookup = members.set_index("identifier")
        for identifier in identifiers:
            scan = f"lidc_{lookup.loc[identifier, 'scan_id']}"
            windows = np.flatnonzero(scans == scan)
            if not len(windows):
                continue
            header = {key: value[windows[0]] for key, value in headers.items()}
            masks = nodules[nodules.scan_id == lookup.loc[identifier, "scan_id"]]["mask"].tolist()
            items = [(int(window), centres.get((identifier, window), -1), lookup.loc[identifier, "mask"]) for window in windows]
            tasks.append((str(path), items, header, masks, {"volume_masks": []}, None))
    return tasks


def geometry(dataset, workers, smoke):
    output = output_directory(dataset, "geometry", smoke)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    batches = input_batches(dataset, smoke)
    for number, path in enumerate(batches):
        target = output / path.name
        if target.exists():
            continue
        tasks = geometry_tasks(dataset, path, smoke)
        windows, volumes = {}, {}
        with ProcessPoolExecutor(workers) as pool:
            for results, part in pool.map(scan_geometry, tasks):
                volumes.update(part)
                for window, values, scalars in results:
                    windows[window] = (values, scalars)
        order = sorted(windows)
        assert order == list(range(len(order)))
        temporary = target.with_suffix(".tmp")
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update({"source_sha256": sha256(path), "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp()})
            for key in windows[0][0]:
                handle.create_dataset(f"tokens/{key}", data=np.stack([windows[window][0][key] for window in order]))
            scalars = pd.DataFrame([windows[window][1] for window in order])
            for key in scalars:
                handle.create_dataset(f"windows/{key}", data=scalars[key].to_numpy())
            handle.create_dataset("records/identifier", data=np.array(list(volumes), dtype="S"))
            handle.create_dataset("records/volume_mm3", data=np.array(list(volumes.values())))
        temporary.replace(target)
        elapsed = time.monotonic() - started
        print(f"geometry {dataset}: {path.name} ({number + 1}/{len(batches)}), {len(order)} windows, failed lungs {int(scalars.lung_failed.sum())}, {elapsed / 60:.1f} min, "
              f"eta {elapsed / (number + 1) * (len(batches) - number - 1) / 60:.1f} min", flush=True)


# Local: step 1, which concepts the text space reads at the scan and token levels.

def residualize(values, covariate):
    values = values - values.mean(-1, keepdims=True)
    covariate = covariate - covariate.mean(-1, keepdims=True)
    return values - (values * covariate).sum(-1, keepdims=True) / (covariate * covariate).sum(-1, keepdims=True) * covariate


def pearson(first, second):
    first = first - first.mean(-1, keepdims=True)
    second = second - second.mean(-1, keepdims=True)
    return (first * second).sum(-1) / np.sqrt((first * first).sum(-1) * (second * second).sum(-1))


def correlation(score, proxy, adjust=None):
    # (partial) Spearman along the last axis, as vlm_text_check.correlation
    score, proxy = rankdata(score, axis=-1), rankdata(proxy, axis=-1)
    if adjust is not None:
        adjust = rankdata(adjust, axis=-1)
        score, proxy = residualize(score, adjust), residualize(proxy, adjust)
    return pearson(score, proxy)


def contrast(local, target, reference):
    if target.sum() < 1 or reference.sum() < 20:
        return np.full(local.shape[1], np.nan)
    return (local[target].mean(0) - local[reference].mean(0)) / local[reference].std(0)


def within(local, tokens, proxy):
    if tokens.sum() < 20 or np.ptp(proxy[tokens]) == 0:
        return np.full(local.shape[1], np.nan)
    return correlation(local[tokens].T, proxy[tokens][None])


def token_statistics(local, tokens, scalars):
    missing = np.full(local.shape[1], np.nan)
    healthy = not bool(scalars["lung_failed"])
    fraction = tokens["lung_fraction"]
    lung = (fraction >= 0.9) & healthy
    centre = int(tokens["centre_token"])
    nodule = tokens["distance_member"] <= 4.0
    if centre >= 0:
        nodule[centre] = True
    result = {"nodule_mean": local[nodule].mean(0) if centre >= 0 else missing,
              "contrast_nodule_tokens": contrast(local, nodule, lung & (tokens["distance_any"] > 32.0)) if centre >= 0 else missing,
              "within_token_laa": within(local, lung, tokens["laa"]), "within_token_haa": within(local, lung, tokens["haa"]), "within_height": within(local, lung, tokens["height"]),
              "within_medial": within(local, tokens["band"] & healthy, tokens["medial"]), "within_boundary": within(local, (fraction > 0) & healthy, -np.abs(fraction - 0.5)),
              "contrast_metal_tokens": contrast(local, tokens["metal"] >= 8.0, tokens["metal_reference"]) if scalars["metal_mm3"] >= configuration()["geometry"]["minimum_metal_mm3"] else missing}
    counts = {"nodule_tokens": int(nodule.sum()) if centre >= 0 else 0, "lung_tokens": int(lung.sum()), "band_tokens": int((tokens["band"] & healthy).sum()), "metal_tokens": int((tokens["metal"] >= 8.0).sum())}
    return result, counts


def window_lookup(dataset):
    lookup = {}
    for path in sorted(output_directory(dataset, "encoded", False).glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            for index, (scan, official) in enumerate(zip(handle["scan"][:].astype(str), handle["official"][:])):
                if official:
                    assert scan not in lookup, scan
                    lookup[scan] = (path.name, index)
    return lookup


def patient_table():
    spec = configuration()
    storage = context()[0]
    members, cases = luna_members(), luna_cases()
    rows = cases.set_index("identifier").loc[members.identifier]
    assert (rows.split.to_numpy() == members.split.to_numpy()).all() and (rows.PatientID.to_numpy() == members.PatientID.to_numpy()).all()
    radiomics = pd.read_csv(Path(storage["runs"]) / spec["radiomics_run"] / "radiomics_features.csv").set_index("identifier").loc[members.identifier]
    volumes = {}
    for path in sorted(output_directory("luna25", "geometry", False).glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            volumes.update(zip(handle["records/identifier"][:].astype(str), handle["records/volume_mm3"][:]))
    assert set(volumes) == set(cases.identifier)
    cases["log_record_volume"] = np.log(cases.identifier.map(volumes))
    by_scan = cases.groupby("SeriesInstanceUID").agg(log_max_volume=("log_record_volume", "max"), nodule_count=("identifier", "size"))
    luna = pd.DataFrame({"dataset": "luna25", "split": members.split.to_numpy(), "PatientID": members.PatientID.to_numpy(), "identifier": members.identifier.to_numpy(),
                         "scan": rows.SeriesInstanceUID.to_numpy(), "log_volume": np.log(radiomics["nodule__shape_VoxelVolume"].to_numpy()), "mean_hu": radiomics["nodule__firstorder_Mean"].to_numpy(),
                         "p90_hu": radiomics["nodule__firstorder_90Percentile"].to_numpy(), "lung_boundary_contact": radiomics["contact__lung_boundary_contact_fraction"].to_numpy(),
                         "sphericity": radiomics["nodule__shape_Sphericity"].to_numpy(), "malignant": members.label.to_numpy().astype(float)})
    luna = luna.join(by_scan, on="scan")
    lidc_members, _ = lidc_tables()
    lidc = pd.DataFrame({"dataset": "lidc", "split": lidc_members.split.to_numpy(), "PatientID": lidc_members.PatientID.to_numpy(), "identifier": lidc_members.identifier.to_numpy(),
                         "scan": [f"lidc_{value}" for value in lidc_members.scan_id], "log_volume": np.log(lidc_members.consensus_volume_mm3.to_numpy()),
                         "lung_boundary_contact": lidc_members.contact_lung_boundary_contact_fraction.to_numpy(), "lidc_malignancy": lidc_members.malignancy.to_numpy(),
                         "lidc_calcification": lidc_members.calcification.to_numpy(), "lidc_spiculation": lidc_members.spiculation.to_numpy(), "lidc_lobulation": lidc_members.lobulation.to_numpy(),
                         "lidc_margin": lidc_members.margin.to_numpy(), "lidc_texture": lidc_members.texture.to_numpy(), "lidc_part_solid": 1 - np.abs(lidc_members.texture.to_numpy() - 3) / 2})
    lidc["log_max_volume"] = lidc.log_volume
    return pd.concat([luna, lidc], ignore_index=True)


def gather(patients, concepts, controls=None):
    # controls: indices of the control concepts when every score is taken relative to their mean (specific_variant)
    statistics = {name: np.full((len(patients), concepts), np.nan) for name in ["nodule_mean", "contrast_nodule_tokens", "within_token_laa", "within_token_haa", "within_height", "within_medial", "within_boundary", "contrast_metal_tokens"]}
    scores = np.full((len(patients), concepts), np.nan)
    scalars, counts = [{} for _ in range(len(patients))], [{} for _ in range(len(patients))]
    for dataset in ["luna25", "lidc"]:
        lookup = window_lookup(dataset)
        chosen = patients.index[patients.dataset == dataset]
        assert all(scan in lookup for scan in patients.loc[chosen, "scan"]), dataset
        for name in sorted({lookup[scan][0] for scan in patients.loc[chosen, "scan"]}):
            with h5py.File(output_directory(dataset, "encoded", False) / name, "r") as encoded, h5py.File(output_directory(dataset, "geometry", False) / name, "r") as geometry:
                assert encoded.attrs["source_sha256"] == geometry.attrs["source_sha256"], name
                window_scalars = pd.DataFrame({key: geometry[f"windows/{key}"][:] for key in geometry["windows"]})
                for row in [row for row in chosen if lookup[patients.loc[row, "scan"]][0] == name]:
                    index = lookup[patients.loc[row, "scan"]][1]
                    scores[row] = encoded["scan_score"][index]
                    local = encoded["local"][index].astype(np.float32)
                    if controls is not None:
                        scores[row] -= scores[row][controls].mean()
                        local -= local[:, controls].mean(1, keepdims=True)
                    tokens = {key: geometry[f"tokens/{key}"][index] for key in geometry["tokens"]}
                    scalars[row] = window_scalars.iloc[index].to_dict()
                    values, counts[row] = token_statistics(local, tokens, scalars[row])
                    for key, value in values.items():
                        statistics[key][row] = value
        print(f"gather: {dataset} {len(chosen)} patients", flush=True)
    scalars = pd.DataFrame(scalars)
    patients = patients.assign(pd15=scalars.pd15.to_numpy(), haa=scalars.haa_fraction.to_numpy(), log_metal=np.log1p(scalars.metal_mm3.to_numpy()), noise=scalars.noise_hu.to_numpy(), lung_failed=scalars.lung_failed.to_numpy(),
                               lung_litres=scalars.lung_litres.to_numpy(), metal_mm3=scalars.metal_mm3.to_numpy())
    return patients.join(pd.DataFrame(counts)), scores, statistics


def test_values(entry, level, scores, statistics):
    if level == "scan":
        return scores
    if entry["kind"] == "across":
        return statistics["nodule_mean"]
    if entry["kind"] == "contrast":
        return statistics[f"contrast_{entry['target']}"]
    return statistics[f"within_{entry['proxy']}"]


def evaluate(patients, scores, statistics, words):
    spec = configuration()
    concepts = list(words["concepts"])
    groups = dict(zip(concepts, words["groups"]))
    controls = [index for index, concept in enumerate(concepts) if groups[concept] == "control"]
    generator = np.random.default_rng(spec["seed"])
    rows = []
    for level, entries in spec["tests"].items():
        for number, entry in enumerate(entries):
            kind = entry.get("kind", "scan")
            matrix = test_values(entry, level, scores, statistics)
            across = level == "scan" or kind == "across"
            for dataset in entry["datasets"]:
                for split in spec["statistics"]["splits"][dataset]:
                    chosen = (patients.dataset == dataset).to_numpy() & (patients.split == split).to_numpy() & np.isfinite(matrix).all(1)
                    proxy = patients[entry["proxy"]].to_numpy(dtype=float) if across else None
                    adjust = patients[entry["adjust"]].to_numpy(dtype=float) if across and entry.get("adjust") else None
                    if across:
                        chosen &= np.isfinite(proxy) & (np.isfinite(adjust) if adjust is not None else True)
                    selected = np.flatnonzero(chosen)
                    draws = generator.integers(0, len(selected), (spec["bootstrap"], len(selected))) if len(selected) else None

                    def statistic(column, rows_index=None):
                        values = matrix[selected, column] if rows_index is None else matrix[selected[rows_index], column]
                        if not across:
                            return values.mean(-1)
                        pick = selected if rows_index is None else selected[rows_index]
                        return correlation(values, proxy[pick], adjust[pick] if adjust is not None else None)

                    for concept in entry["concepts"]:
                        column = concepts.index(concept)
                        row = {"level": level, "test": f"{level}_{number:02d}", "kind": kind, "concept": concept, "group": groups[concept], "dataset": dataset, "split": split,
                               "proxy": entry.get("proxy", entry.get("target")), "adjust": entry.get("adjust"), "sign": entry["sign"], "patients": len(selected)}
                        if len(selected) < spec["statistics"]["minimum_patients"]:
                            rows.append({**row, "passes": False, "note": "fewer than the minimum patients"})
                            continue
                        value = float(statistic(column))
                        boot = statistic(column, draws)
                        low, high = np.quantile(boot, [0.025, 0.975])
                        control_values = [float(statistic(index)) for index in controls]
                        bar = float(np.max(np.abs(control_values)))
                        excludes = low > 0 if entry["sign"] > 0 else high < 0
                        rows.append({**row, "statistic": value, "ci_low": float(low), "ci_high": float(high), "aligned": entry["sign"] * value, "control_bar": bar,
                                     "control_values": ";".join(f"{item:+.3f}" for item in control_values), "passes": bool(entry["sign"] * value > 0 and excludes and abs(value) > bar), "note": ""})
                        print(f"{level:5s} {row['test']} {concept:32s} {dataset:6s} {split:12s} n {len(selected):4d} {value:+.3f} [{low:+.3f}, {high:+.3f}] bar {bar:.3f} {'PASS' if rows[-1]['passes'] else ''}", flush=True)
    return pd.DataFrame(rows)


def usability(table, words):
    tests = table.groupby(["level", "test", "concept", "dataset"], sort=False).agg(passes=("passes", "all"), splits=("split", "size")).reset_index()
    rows = []
    for concept, group in zip(words["concepts"], words["groups"]):
        row = {"concept": concept, "group": group}
        for level in ["scan", "token"]:
            chosen = tests[(tests.concept == concept) & (tests.level == level)]
            row[f"{level}_tests"] = len(chosen)
            row[f"{level}_passed"] = int(chosen.passes.sum())
            row[level] = "untested" if not len(chosen) else ("usable" if chosen.passes.all() else "not usable")
        rows.append(row)
    return tests, pd.DataFrame(rows)


def calibrate(variant):
    spec = configuration()
    output = run_root() / ("calibration" if variant == "primary" else "calibration_specific")
    output.mkdir(parents=True, exist_ok=True)
    words = load_vocabulary()
    patients = patient_table()
    controls = [index for index, group in enumerate(words["groups"]) if group == "control"] if variant == "specific" else None
    patients, scores, statistics = gather(patients, len(words["concepts"]), controls)
    np.savez(output / "patients.npz", scores=scores, identifiers=patients.identifier.to_numpy(dtype=str), concepts=words["concepts"], **statistics)
    patients.to_csv(output / "patients.csv", index=False)
    table = evaluate(patients, scores, statistics, words)
    table.to_csv(output / "tests.csv", index=False)
    tests, concepts = usability(table, words)
    tests.to_csv(output / "test_verdicts.csv", index=False)
    concepts.to_csv(output / "concepts.csv", index=False)
    counts = {dataset: {split: int(((patients.dataset == dataset) & (patients.split == split)).sum()) for split in spec["statistics"]["splits"][dataset]} for dataset in ["luna25", "lidc"]}
    excluded = {dataset: {"nodule_centre_outside_window": int(((patients.dataset == dataset) & (patients.nodule_tokens == 0)).sum()), "lung_failed": int(((patients.dataset == dataset) & patients.lung_failed).sum()),
                          "metal_at_least_minimum": int(((patients.dataset == dataset) & (patients.metal_mm3 >= spec["geometry"]["minimum_metal_mm3"])).sum())} for dataset in ["luna25", "lidc"]}
    write_json(output / "summary.json", {"run": spec["run_id"], "variant": variant, "patients": counts, "exclusions": excluded, "pass_rule": spec["pass_rule"], "usability_rule": spec["usability_rule"],
                                         "usable": {level: concepts.loc[concepts[level] == "usable", "concept"].tolist() for level in ["scan", "token"]},
                                         "tests": {"total": len(tests), "passed": int(tests.passes.sum())}, "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    for level in ["scan", "token"]:
        figure(table, level, output, variant)
    print(concepts.to_string(index=False), flush=True)


def figures():
    # redraw from the saved test tables; no statistic is recomputed
    for variant, directory in [("primary", "calibration"), ("specific", "calibration_specific")]:
        output = run_root() / directory
        if (output / "tests.csv").exists():
            table = pd.read_csv(output / "tests.csv")
            for level in ["scan", "token"]:
                figure(table, level, output, variant)


def figure(table, level, output, variant):
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dd"
    colours = {"train": "#2563a8", "development": "#c2410c", "discovery": "#2563a8", "confirmation": "#c2410c"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": ink})
    chosen = table[(table.level == level) & (table.group != "control") & table.statistic.notna()]
    keys = list(dict.fromkeys(zip(chosen.test, chosen.concept, chosen.dataset)))
    fig, axis = plt.subplots(figsize=(8.6, 0.24 * len(keys) + 1.4), facecolor=surface)
    axis.set_facecolor(surface)
    labels = []
    for position, (test, concept, dataset) in enumerate(keys):
        rows = chosen[(chosen.test == test) & (chosen.concept == concept) & (chosen.dataset == dataset)]
        bar = rows.control_bar.max()
        axis.add_patch(plt.Rectangle((-bar, position - 0.38), 2 * bar, 0.76, color=grid, lw=0, zorder=1))
        for offset, (_, row) in zip([-0.15, 0.15], rows.iterrows()):
            low, high = sorted([row.sign * row.ci_low, row.sign * row.ci_high])
            axis.plot([low, high], [position + offset] * 2, color=colours[row.split], lw=1.2, zorder=2)
            axis.scatter([row.aligned], [position + offset], s=22, marker="o", facecolor=colours[row.split] if row.passes else surface, edgecolor=colours[row.split], lw=1.2, zorder=3)
        verdict = "passes" if rows.passes.all() else ""
        first = rows.iloc[0]
        target = first.proxy + (f" | {first.adjust}" if isinstance(first.adjust, str) else "")
        labels.append(f"{concept} ~ {target} ({dataset}) {verdict}")
    axis.set_yticks(range(len(keys)), labels)
    axis.set_ylim(len(keys) - 0.5, -0.5)
    axis.axvline(0, color=muted, lw=0.8, zorder=1)
    axis.grid(axis="x", color=grid, lw=0.6)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    unit = "Spearman / mean within-scan Spearman / standardised contrast" if level == "token" else "Spearman (partial where adjusted)"
    axis.set_xlabel(f"statistic in the expected direction ({unit})", color=muted)
    handles = [plt.Line2D([], [], marker="o", color=colour, lw=1.2, markerfacecolor=colour, label=label) for label, colour in [("train or LIDC discovery half", "#2563a8"), ("development or LIDC confirmation half", "#c2410c")]]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=grid, label="largest control concept (|statistic|)"))
    axis.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=7, borderaxespad=0.2)
    scoring = "" if variant == "primary" else ", scores relative to the mean control score"
    axis.set_title(f"FlexiCT-3D-VLM text readout on whole-CT windows, {level} level{scoring}\nfilled marker: the split passes (sign, 95% interval, above every control); 'passes' marks a test passing on both splits", loc="left", fontsize=8.5, pad=20)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"{level}_tests.{suffix}", dpi=170, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


# Local GPU: field-site tokens (f32_b10 FlexiCT-3D, f40_b08 FlexiCT-3D-VLM) of every train and development record.

FIELD_SITES = {"f32_b10": ("flexict_3d", "f32", 10), "f40_b08": ("flexict_vlm", "f40", 8)}


@torch.inference_mode()
def fields():
    from .flexict_luna25 import encode as forward, field_input
    from .models import load_flexict

    pipeline = read_json(ROOT / "configs/encoder_pipeline.json")["models"]
    output = run_root() / "fields"
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = luna_cases()
    for site, (model_name, view, block) in FIELD_SITES.items():
        target = output / f"{site}.h5"
        if target.exists():
            continue
        model, _ = load_flexict(pipeline[model_name]["checkpoint"])
        model.cuda()
        identifiers, tokens, sources, started = [], [], {}, time.monotonic()
        for path in input_batches("luna25", False):
            with h5py.File(path, "r") as handle:
                names = handle["identifier"][:].astype(str)
                for index, identifier in enumerate(names):
                    (patch, _), = forward(model, field_input(handle[view][index]), [block])
                    tokens.append(patch.float().cpu().numpy())
                    identifiers.append(identifier)
            sources[path.name] = sha256(path)
            print(f"fields {site}: {path.name}, {len(identifiers)} records, {(time.monotonic() - started) / 60:.1f} min", flush=True)
        assert set(identifiers) == set(cases.identifier)
        temporary = target.with_suffix(".tmp")
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update({"model": model_name, "checkpoint": pipeline[model_name]["checkpoint"], "view": view, "block": block, "sources": json.dumps(sources),
                                 "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp()})
            handle.create_dataset("identifier", data=np.array(identifiers, dtype="S"))
            handle.create_dataset("tokens", data=np.stack(tokens), chunks=(1, *tokens[0].shape))
        temporary.replace(target)
        del model
        torch.cuda.empty_cache()


# Local: step 2, text names for the stable w20_b16 features.

def split_summaries(patients, split, concepts):
    # per seed sums over the split's member windows, the window sums for the scan route, and token-score moments
    lookup = window_lookup("luna25")
    chosen = patients[(patients.dataset == "luna25") & (patients.split == split)]
    by_file = {}
    for row, scan in zip(chosen.index, chosen.scan):
        by_file.setdefault(lookup[scan][0], []).append((row, lookup[scan][1]))
    totals = {"sum": 0, "square": 0, "weighted": 0, "active": 0}
    window_sums, scores, joints, moments, order = [], [], [], np.zeros((2, concepts)), []
    for name, items in sorted(by_file.items()):
        with h5py.File(output_directory("luna25", "encoded", False) / name, "r") as handle:
            for row, index in items:
                summed = handle["seed_sum"][index].astype(np.float64)
                totals["sum"] = totals["sum"] + summed
                totals["square"] = totals["square"] + handle["seed_square"][index].astype(np.float64)
                totals["weighted"] = totals["weighted"] + handle["seed_weighted"][index].astype(np.float64)
                totals["active"] = totals["active"] + (handle["seed_count"][index] > 0)
                local = handle["local"][index].astype(np.float64)
                moments += [local.sum(0), np.square(local).sum(0)]
                window_sums.append(summed.astype(np.float32))
                scores.append(handle["scan_score"][index])
                joints.append(np.concatenate([handle["cls"][index], handle["mean"][index]]))
                order.append(row)
    count = len(order) * TOKENS
    mean = moments[0] / count
    return {"totals": totals, "window_sums": np.stack(window_sums), "scores": np.stack(scores), "joints": np.stack(joints), "rows": np.array(order),
            "token_mean": mean, "token_std": np.sqrt(moments[1] / count - mean ** 2), "windows": len(order)}


def ablated_scores(joints, projection, direction, window_sums, write, scale):
    # exact scan scores after removing a feature from every token: the mean patch token loses sum_t a_f(t) scale W_dec[f] / 8000
    joint = joints @ projection.T
    changed = joint[:, None, :] - window_sums[:, :, None] * scale / TOKENS * write[None]
    changed = changed / np.linalg.norm(changed, axis=-1, keepdims=True)
    return changed @ direction.T


def feature_rule(D, colocation, active, controls, column, percentile):
    # direction and colocation conditions of naming.token_route for one concept column; colocation holds (train, development)
    direction_ok = (D[:, column] >= percentile) & (D[:, column] > D[:, controls].max(1))
    located = np.all([(value[:, column] > 0) & (value[:, column] > np.abs(value[:, controls]).max(1)) for value in colocation], 0)
    return direction_ok & located & np.all([count >= 30 for count in active], 0)


def name_features():
    from .models import published
    from .sae import SpatialDictionary

    spec = configuration()
    output = run_root() / "naming"
    output.mkdir(parents=True, exist_ok=True)
    words = load_vocabulary()
    concepts, groups = list(words["concepts"]), list(words["groups"])
    controls = [index for index, group in enumerate(groups) if group == "control"]
    usable = pd.read_csv(run_root() / "calibration" / "concepts.csv")
    token_concepts = usable.concept[usable.token == "usable"].tolist()
    scan_concepts = usable.concept[(usable.scan == "usable") & (usable.token != "usable")].tolist()
    sae_root, entries = production_dictionaries()
    with np.load(sae_root / "stability" / spec["sae"]["group"] / "features.npz") as saved:
        stability = {key: saved[key] for key in saved.files}
    assert stability["seeds"].tolist() == [entry["seed"] for entry in entries]
    stable = stability["development_live"][0] & (stability["reappearance"][0] >= 6)
    projection = torch.load(Path(published("flexict")["directory"]) / "ct_3d_vlm.pth", map_location="cpu", weights_only=True)["model"]["vlm_vision_projection.weight"].double().numpy()
    direction = words["direction"].astype(np.float64)
    seeds = []
    for entry in entries:
        dictionary = SpatialDictionary(sae_root / entry["path"], device="cpu")
        write = dictionary.weights["W_dec"].double().numpy() @ projection[:, 864:].T
        scale = float(dictionary.scale)
        cosine = (write / np.linalg.norm(write, axis=1, keepdims=True)) @ (direction / np.linalg.norm(direction, axis=1, keepdims=True)).T
        seeds.append({"write": write, "scale": scale, "D": cosine, "g": scale * write @ direction.T})
    patients = pd.read_csv(run_root() / "calibration" / "patients.csv", dtype={"PatientID": str})
    splits = {split: split_summaries(patients, split, len(concepts)) for split in ["train", "development"]}
    print("naming: windows", {split: value["windows"] for split, value in splits.items()}, flush=True)
    colocation, active = [], []
    for seed, item in enumerate(seeds):
        values, counts = [], []
        for split, summary in splits.items():
            totals = summary["totals"]
            with np.errstate(invalid="ignore", divide="ignore"):
                located = (totals["weighted"][seed] - item["g"] * totals["square"][seed][:, None]) / totals["sum"][seed][:, None]
            values.append(np.nan_to_num((located - summary["token_mean"]) / summary["token_std"], nan=-np.inf))
            counts.append(totals["active"][seed])
        colocation.append(values)
        active.append(counts)
    train_live = [splits["train"]["totals"]["sum"][seed] > 0 for seed in range(len(seeds))]
    percentiles = [np.percentile(item["D"][train_live[seed]], 99, axis=0) for seed, item in enumerate(seeds)]
    passed = {}
    for concept in token_concepts:
        column = concepts.index(concept)
        passed[concept] = [feature_rule(item["D"], colocation[seed], active[seed], controls, column, percentiles[seed][column]) for seed, item in enumerate(seeds)]
    for concept in scan_concepts:
        column = concepts.index(concept)
        result = []
        for seed, item in enumerate(seeds):
            direction_ok = (item["D"][:, column] >= percentiles[seed][column]) & (item["D"][:, column] > item["D"][:, controls].max(1)) & np.all([count >= 30 for count in active[seed]], 0)
            ok = np.zeros(len(direction_ok), dtype=bool)
            for feature in np.flatnonzero(direction_ok):
                checks = []
                for split, summary in splits.items():
                    sums = summary["window_sums"][:, seed, feature]
                    ablated = ablated_scores(summary["joints"], projection, direction, sums[:, None], item["write"][[feature]], item["scale"])[:, 0]
                    association = correlation(sums, ablated[:, column])
                    control_bar = max(abs(correlation(sums, ablated[:, index])) for index in controls)
                    checks.append(association > 0 and association > control_bar)
                ok[feature] = all(checks)
            result.append(ok)
        passed[concept] = result
    matched, cosine = stability["matched"][0], stability["best_cosine"][0]
    rows = []
    train = splits["train"]
    reference = seeds[0]
    deviation = train["scores"].astype(np.float64).std(0)
    for feature in np.flatnonzero(stable):
        names, routes, agreement = [], [], {}
        for concept, results in passed.items():
            others = [cosine[seed, feature] >= 0.7 and bool(results[seed][matched[seed, feature]]) for seed in range(1, len(seeds))]
            agreement[concept] = int(sum(others))
            if results[0][feature] and sum(others) >= 6:
                names.append(concept)
                routes.append("token" if concept in token_concepts else "scan")
        best = int(np.argmax(np.where(np.isin(np.arange(len(concepts)), controls), -np.inf, reference["D"][feature])))
        sums = train["window_sums"][:, 0, feature]
        on = sums > 0
        effect = ((ablated_scores(train["joints"][on], projection, direction, sums[on, None], reference["write"][[feature]], reference["scale"])[:, 0] - train["scores"][on]).mean(0) / deviation) if on.any() else np.full(len(concepts), np.nan)
        order = sorted(names, key=lambda concept: -reference["D"][feature, concepts.index(concept)])
        rows.append({"feature": int(feature), "name": order[0] if order else "", "names": ";".join(order), "routes": ";".join(routes[names.index(concept)] for concept in order),
                     "best_direction_concept": concepts[best], "best_direction": float(reference["D"][feature, best]), "control_direction_max": float(reference["D"][feature, controls].max()),
                     "train_windows_active": int(active[0][0][feature]), "development_windows_active": int(active[0][1][feature]),
                     **{f"direction_{concept}": float(reference["D"][feature, concepts.index(concept)]) for concept in passed},
                     **{f"colocation_train_{concept}": float(colocation[0][0][feature, concepts.index(concept)]) for concept in token_concepts},
                     **{f"colocation_development_{concept}": float(colocation[0][1][feature, concepts.index(concept)]) for concept in token_concepts},
                     **{f"seeds_agreeing_{concept}": agreement[concept] for concept in passed},
                     **{f"causal_{concept}": float(effect[concepts.index(concept)]) for concept in passed}})
    table = pd.DataFrame(rows)
    table.to_csv(output / "features.csv", index=False)
    named = table[table.name != ""]
    write_json(output / "summary.json", {"run": spec["run_id"], "group": spec["sae"]["group"], "stable_features": int(stable.sum()), "token_concepts": token_concepts, "scan_concepts": scan_concepts,
                                         "named_features": len(named), "names": named.name.value_counts().to_dict(),
                                         "reference_passing_before_cross_seed": {concept: int(results[0][stable].sum()) for concept, results in passed.items()},
                                         "windows": {split: value["windows"] for split, value in splits.items()}, "rule": spec["naming"],
                                         "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    print(f"naming: {len(named)} of {int(stable.sum())} stable features named", named.name.value_counts().to_dict(), flush=True)


# Local: step 5, names for the field-site features through w20_b16.

def rank_columns(values):
    ranked = rankdata(values, axis=0)
    ranked = ranked - ranked.mean(0)
    norm = np.sqrt((ranked * ranked).sum(0))
    return ranked / np.where(norm > 0, norm, np.inf)


@torch.inference_mode()
def site_activations(site, identifiers, reduce):
    from .sae import SpatialDictionary

    group = site_group(site)
    entry = group["dictionaries"][0]
    sae_root = Path(context()[0]["runs"]) / configuration()["sae_campaign"]
    assert sha256(sae_root / entry["path"]) == entry["sha256"]
    dictionary = SpatialDictionary(sae_root / entry["path"])
    tokens = site_tokens(site, identifiers)
    values = torch.stack([(dictionary.encode(x.cuda()).sum(0) if reduce == "sum" else dictionary.encode(x.cuda()).amax(0)) for x in tokens]).cpu().numpy()
    with np.load(sae_root / "stability" / group["group"] / "features.npz") as saved:
        stable = saved["development_live"][0] & (saved["reappearance"][0] >= 6)
    return values, stable


def bridge():
    spec = configuration()
    output = run_root() / "bridge"
    output.mkdir(parents=True, exist_ok=True)
    words = load_vocabulary()
    concepts = list(words["concepts"])
    controls = [index for index, group in enumerate(words["groups"]) if group == "control"]
    usable = pd.read_csv(run_root() / "calibration" / "concepts.csv")
    token_concepts = [concepts.index(concept) for concept in usable.concept[usable.token == "usable"]]
    patients = pd.read_csv(run_root() / "calibration" / "patients.csv", dtype={"PatientID": str})
    with np.load(run_root() / "calibration" / "patients.npz") as saved:
        assert (saved["identifiers"] == patients.identifier.to_numpy(dtype=str)).all()
        nodule_mean = saved["nodule_mean"]
    chosen = (patients.dataset == "luna25").to_numpy()
    members, nodule_mean = patients[chosen].reset_index(drop=True), nodule_mean[chosen]
    w20, w20_stable = site_activations("w20_b16", members.identifier.tolist(), "sum")
    w20_names = site_names("w20_b16")
    w20_features = np.flatnonzero(w20_stable)
    named = np.array([bool(w20_names.get(feature, "")) for feature in w20_features])
    summary = {}
    for site in spec["bridge"]["sites"]:
        values, stable = site_activations(site, members.identifier.tolist(), "max")
        features = np.flatnonzero(stable)
        rows = []
        correlations, direct = {}, {}
        for split in ["train", "development"]:
            part = (members.split == split).to_numpy()
            field = rank_columns(values[part][:, features])
            correlations[split] = field.T @ rank_columns(w20[part][:, w20_features])
            located = part & np.isfinite(nodule_mean).all(1)
            direct[split] = rank_columns(values[located][:, features]).T @ rank_columns(nodule_mean[located])
        for position, feature in enumerate(features):
            train, development = correlations["train"][position], correlations["development"][position]
            best = int(np.argmax(train))
            best_named = int(np.flatnonzero(named)[np.argmax(train[named])]) if named.any() else None
            bridged = best_named is not None and train[best_named] >= 0.5 and development[best_named] >= 0.5
            inherited = w20_names[w20_features[best_named]] if bridged else ""
            holds = {concepts[index]: bool(all(direct[split][position, index] > 0 and direct[split][position, index] > np.abs(direct[split][position, controls]).max() for split in direct)) for index in token_concepts}
            rows.append({"feature": int(feature), "name": inherited if inherited and holds.get(inherited, False) else "",
                         "best_w20_feature": int(w20_features[best]), "best_w20_name": w20_names.get(int(w20_features[best]), ""), "best_w20_train": float(train[best]), "best_w20_development": float(development[best]),
                         "best_named_w20_feature": int(w20_features[best_named]) if best_named is not None else -1, "best_named_w20_name": w20_names[w20_features[best_named]] if best_named is not None else "",
                         "best_named_train": float(train[best_named]) if best_named is not None else np.nan, "best_named_development": float(development[best_named]) if best_named is not None else np.nan,
                         "bridge_holds": bridged, "direct_holds": ";".join(concept for concept, value in holds.items() if value),
                         **{f"direct_{split}_{concepts[index]}": float(direct[split][position, index]) for split in direct for index in token_concepts}})
        table = pd.DataFrame(rows)
        table.to_csv(output / f"{site}.csv", index=False)
        summary[site] = {"stable_features": len(features), "bridged": int(table.bridge_holds.sum()), "direct_any": int((table.direct_holds != "").sum()), "named": int((table.name != "").sum()),
                         "names": table.name[table.name != ""].value_counts().to_dict(), "best_w20_train_median": float(table.best_w20_train.median())}
        print("bridge", site, summary[site], flush=True)
    write_json(output / "summary.json", {**summary, "w20_named_stable": int(named.sum()), "records": members.split.value_counts().to_dict(), "rule": spec["bridge"],
                                         "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})


# Local: step 4, which features the three malignancy classifiers use.

def site_group(site):
    spec = configuration()
    return read_json(Path(context()[0]["runs"]) / spec["sae_campaign"] / "report" / "production.json")["sites"][spec["contributions"]["sites"][site]]


def load_head(site):
    from .frozen_heads import SpatialHead

    spec = configuration()
    storage = context()[0]
    model, name = spec["contributions"]["sites"][site].split(":")
    remote = read_json(Path(storage["runs"]) / spec["sae_campaign"] / "evaluation" / f"{site_group(site)['group']}.json")["classifier"]["head_run"]
    run = Path(storage["runs"]) / spec["inputs_suite"] / "mayo" / "heads" / model / name / Path(remote).name
    head = SpatialHead(864, "max", np.zeros(864, dtype=np.float32), np.ones(864, dtype=np.float32)).cuda()
    head.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    return head.eval(), run


def site_tokens(site, identifiers):
    values = {}
    if site == "w20_b16":
        for path in sorted(output_directory("luna25", "encoded", False).glob("batch_*.h5")):
            with h5py.File(path, "r") as handle:
                names = handle["readout/identifier"][:].astype(str)
                keep = np.flatnonzero(np.isin(names, identifiers))
                if len(keep):
                    data = handle["readout/tokens"][:]
                    values.update({names[index]: data[index] for index in keep})
    else:
        with h5py.File(run_root() / "fields" / f"{site}.h5", "r") as handle:
            names = handle["identifier"][:].astype(str)
            for index in np.flatnonzero(np.isin(names, identifiers)):
                values[names[index]] = handle["tokens"][index]
    assert set(values) == set(identifiers), site
    return torch.from_numpy(np.stack([values[identifier] for identifier in identifiers]))


def site_names(site):
    path = run_root() / "naming" / "features.csv" if site == "w20_b16" else run_root() / "bridge" / f"{site}.csv"
    table = pd.read_csv(path).fillna({"name": ""})
    return dict(zip(table.feature, table.name))


@torch.inference_mode()
def record_effects(head, dictionary, tokens, batch=128):
    codes = dictionary.encode(tokens)
    active = torch.nonzero(codes.sum(0) > 0).flatten()
    base = head(tokens[None])[0]
    effects = []
    for start in range(0, len(active), batch):
        chosen = active[start:start + batch]
        decoded = codes[:, chosen].T[:, :, None] * (dictionary.weights["W_dec"][chosen] * dictionary.scale)[:, None, :]
        effects.append(base - head(tokens[None] - decoded))
    return active.cpu().numpy(), torch.cat(effects).cpu().numpy(), float(base), codes


@torch.inference_mode()
def contributions():
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .models import published
    from .sae import SpatialDictionary

    spec = configuration()
    output = run_root() / "contributions"
    output.mkdir(parents=True, exist_ok=True)
    sae_root = Path(context()[0]["runs"]) / spec["sae_campaign"]
    cases = luna_cases()
    volumes = {}
    for path in sorted(output_directory("luna25", "geometry", False).glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            volumes.update(zip(handle["records/identifier"][:].astype(str), handle["records/volume_mm3"][:]))
    cases["log_volume"] = np.log(cases.identifier.map(volumes))
    development = cases[cases.split == "development"].sort_values("identifier").reset_index(drop=True)
    summary = {}
    for site in spec["contributions"]["sites"]:
        group = site_group(site)
        entry = group["dictionaries"][0]
        assert entry["seed"] == spec["sae"]["reference_seed"] and sha256(sae_root / entry["path"]) == entry["sha256"]
        dictionary = SpatialDictionary(sae_root / entry["path"])
        with np.load(sae_root / "stability" / group["group"] / "features.npz") as saved:
            stable = saved["development_live"][0] & (saved["reappearance"][0] >= 6)
        names = site_names(site)
        head, run = load_head(site)
        stored = pd.read_csv(run / "development_predictions.csv").set_index("identifier").loc[development.identifier]
        tokens = site_tokens(site, development.identifier.tolist())
        rows, logits, categories = [], [], {"named": [], "unnamed": [], "unstable": []}
        for feature in range(dictionary.weights["W_dec"].shape[0]):
            category = "unstable" if not stable[feature] else ("named" if names.get(feature, "") else "unnamed")
            categories[category].append(feature)
        joint = {category: [] for category in categories}
        decoder = dictionary.weights["W_dec"] * dictionary.scale
        for record in range(len(development)):
            x = tokens[record].cuda()
            active, effect, base, codes = record_effects(head, dictionary, x)
            logits.append(base)
            rows += [(record, int(feature), float(value)) for feature, value in zip(active, effect)]
            for category, members in categories.items():
                chosen = torch.tensor(members, device="cuda", dtype=torch.long)
                joint[category].append(float(head((x - codes[:, chosen] @ decoder[chosen])[None])[0]))
            if (record + 1) % 200 == 0:
                print(f"contributions {site}: {record + 1}/{len(development)} records", flush=True)
        logits = np.array(logits)
        probability = 1 / (1 + np.exp(-logits))
        check = float(np.abs(probability - stored.probability.to_numpy()).max())
        assert check < 5e-3, (site, check)
        effects = pd.DataFrame(rows, columns=["record", "feature", "effect"])
        effects.to_csv(output / f"{site}_effects.csv.gz", index=False)
        table = effects.assign(absolute=effects.effect.abs()).groupby("feature").agg(mean_absolute=("absolute", "sum"), mean_effect=("effect", "sum"), records=("record", "size"))
        table[["mean_absolute", "mean_effect"]] /= len(development)
        table["stable"] = stable[table.index]
        table["name"] = [names.get(feature, "") for feature in table.index]
        size = development.log_volume.to_numpy()
        table["volume_spearman"] = [correlation(part.effect.to_numpy(), size[part.record.to_numpy()]) if len(part) >= 30 else np.nan for _, part in effects.groupby("feature")]
        table = table.sort_values("mean_absolute", ascending=False)
        table.to_csv(output / f"{site}_features.csv")
        labels = development.label.to_numpy()
        groups = {category: {"features": len(members), "summed_mean_absolute_effect": float(table.loc[table.index.isin(members), "mean_absolute"].sum()),
                             "joint_mean_absolute_logit_change": float(np.abs(logits - np.array(joint[category])).mean()), "joint_auroc": float(roc_auc_score(labels, joint[category]))}
                  for category, members in categories.items()}
        summary[site] = {"group": group["group"], "head_run": run.name, "records": len(development), "probability_max_abs_difference": check, "auroc": float(roc_auc_score(labels, logits)),
                         "groups": groups, "top50": table.head(50).reset_index().to_dict("records")}
        print(f"contributions {site}: check {check:.2e}, auroc {summary[site]['auroc']:.4f}", {key: (value["summed_mean_absolute_effect"], value["joint_auroc"]) for key, value in groups.items()}, flush=True)
        if site == "w20_b16":
            words = load_vocabulary()
            usable = pd.read_csv(run_root() / "calibration" / "concepts.csv")
            chosen = [list(words["concepts"]).index(concept) for concept in usable.concept[usable.token == "usable"]]
            projection = torch.load(Path(published("flexict")["directory"]) / "ct_3d_vlm.pth", map_location="cpu", weights_only=True)["model"]["vlm_vision_projection.weight"].float()
            matrix = projection[:, 864:].T @ torch.from_numpy(words["direction"]).T
            train = cases[cases.split == "train"].sort_values("identifier").reset_index(drop=True)
            train_tokens = site_tokens(site, train.identifier.tolist())
            train_logits = np.concatenate([head(part.cuda()).cpu().numpy() for part in train_tokens.split(128)])
            design = {split: (part @ matrix).mean(1).numpy()[:, chosen] for split, part in [("train", train_tokens), ("development", tokens)]}
            models = {"concepts": (design["train"], design["development"]), "log_volume": (train.log_volume.to_numpy()[:, None], size[:, None]),
                      "concepts_and_log_volume": (np.c_[design["train"], train.log_volume.to_numpy()], np.c_[design["development"], size])}
            summary[site]["concept_model"] = {"concepts": [words["concepts"][index] for index in chosen]}
            for key, (fit_x, test_x) in models.items():
                if fit_x.shape[1] == 0:
                    continue
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30), cv=5)).fit(fit_x, train_logits)
                summary[site]["concept_model"][key] = float(r2_score(logits, model.predict(test_x)))
            print("concept model", summary[site]["concept_model"], flush=True)
        torch.cuda.empty_cache()
    write_json(output / "summary.json", {**summary, "rule": spec["contributions"], "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["vocabulary", "lidc_prepare", "encode", "geometry", "calibrate", "figures", "name", "fields", "bridge", "contributions"], required=True)
    parser.add_argument("--dataset", choices=["luna25", "lidc"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--variant", choices=["primary", "specific"], default="primary")
    args = parser.parse_args()
    if args.stage == "vocabulary":
        vocabulary()
    elif args.stage == "lidc_prepare":
        lidc_prepare(args.workers, args.smoke)
    elif args.stage == "encode":
        encode(args.dataset, args.smoke)
    elif args.stage == "geometry":
        geometry(args.dataset, args.workers, args.smoke)
    elif args.stage == "calibrate":
        calibrate(args.variant)
    elif args.stage == "figures":
        figures()
    elif args.stage == "fields":
        fields()
    elif args.stage == "contributions":
        contributions()
    elif args.stage == "bridge":
        bridge()
    else:
        name_features()


if __name__ == "__main__":
    main()

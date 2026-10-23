import argparse
import csv
import gc
import hashlib
import importlib
import io
import multiprocessing
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from filelock import FileLock, Timeout
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .nodule_mask import target_component


CONFIG_PATH = ROOT / "configs/flexict_sae.json"
BANDS = [("inside", -np.inf, 0.0), ("margin_0_4mm", 0.0, 4.0), ("near_4_10mm", 4.0, 10.0)]
PATCH = 8
FIELD_SIZE = 48
WINDOW = 160
RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])


def configuration():
    return read_json(CONFIG_PATH)


def campaign_root():
    return Path(context()[0]["runs"]) / configuration()["campaign"]


def suite_runs():
    return Path(context()[0]["runs"]) / configuration()["features_suite"]


def stable_value(text):
    # same seed derivation as sae_context.stable_value
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def digest(value):
    return hashlib.sha256(repr(sorted(value.items())).encode()).hexdigest()


def site_list():
    return [(model, site, spec) for model, entries in configuration()["models"].items() for site, spec in entries.items()]


def signed_distance(mask, spacing):
    # same transform as sae_context.signed_distance
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing)
    return np.where(mask, -inside, outside).astype(np.float32)


# Local: token-centre distances to the nodule surface.

def field_token_centres(crop_affine, crop_shape, field_mm):
    # physical_field target grid (LPS orientation); token order (S, P, L) as in the flattened readout
    target = np.eye(4)
    target[:3, :3] = np.diag(np.full(3, field_mm / FIELD_SIZE) * np.array([-1, -1, 1]))
    target[:3, 3] = nib.affines.apply_affine(crop_affine, np.asarray(crop_shape) // 2) - target[:3, :3] @ (np.full(3, FIELD_SIZE) // 2)
    grid = np.arange(FIELD_SIZE // PATCH) * PATCH + (PATCH - 1) / 2
    s, p, l = np.meshgrid(grid, grid, grid, indexing="ij")
    return nib.affines.apply_affine(target, np.stack([l, p, s], -1).reshape(-1, 3))


def nodule_surface(mask_path):
    image = nib.load(mask_path)
    component, _ = target_component(np.asarray(image.dataobj))
    boundary = np.argwhere(component & ~ndimage.binary_erosion(component))
    return image, component, cKDTree(nib.affines.apply_affine(image.affine, boundary))


def window_token_centres(scan, start):
    # official 2.0 mm volume geometry (flexict_luna25.official_volume) and its padded MONAI window
    size = np.array([int(round(count * step / 2.0)) for count, step in zip(scan.GetSize(), scan.GetSpacing())])[::-1]
    pad = (np.maximum(size, WINDOW) - size) // 2
    grid = np.arange(WINDOW // PATCH) * PATCH + (PATCH - 1) / 2
    z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")
    index = np.stack([z, y, x], -1).reshape(-1, 3) + np.asarray(start) - pad
    direction = np.asarray(scan.GetDirection()).reshape(3, 3)
    lps = np.asarray(scan.GetOrigin()) + (direction @ (2.0 * index[:, ::-1]).T).T
    return lps * RAS_TO_LPS


def scan_distances(scan_path, rows, windows, fields):
    scan = sitk.ImageFileReader()
    scan.SetFileName(scan_path)
    scan.ReadImageInformation()
    surfaces, records = [], {}
    for row in rows:
        image, component, tree = nodule_surface(row["mask"])
        surfaces.append((image, component, tree))
        distance = signed_distance(component, nib.affines.voxel_sizes(image.affine))
        values = {}
        for view, field_mm in fields.items():
            centres = field_token_centres(image.affine, image.shape, field_mm)
            voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), centres)
            values[view] = ndimage.map_coordinates(distance, voxel.T, order=1, mode="nearest").astype(np.float32)
        records[row["identifier"]] = values
    result = {}
    for scan_name, start in windows:
        centres = window_token_centres(scan, start)
        signed = []
        for image, component, tree in surfaces:
            gap, _ = tree.query(centres)
            voxel = np.rint(nib.affines.apply_affine(np.linalg.inv(image.affine), centres)).astype(int)
            within = np.all((voxel >= 0) & (voxel < np.asarray(component.shape)), 1)
            inside = np.zeros(len(centres), dtype=bool)
            inside[within] = component[tuple(voxel[within].T)]
            signed.append(np.where(inside, -gap, gap))
        # inside any nodule gives a negative value; otherwise the nearest surface
        result[scan_name] = np.min(signed, 0).astype(np.float32)
    return records, result


def distances(workers):
    from .flexict_luna25 import specification as luna_spec

    storage, _, dataset = context()
    spec = configuration()
    inputs = Path(storage["preprocessed"]) / spec["features_suite"] / "inputs"
    cases = pd.read_csv(inputs / "cases.csv", dtype={"PatientID": str})
    cases = cases[cases.split != "test"].copy()
    cases["mask"] = [str(dataset / "labelsTr" / f"{identifier}.nii.gz") for identifier in cases.identifier]
    cases["scan"] = [str(Path(storage["luna25_root"]) / "images" / "luna25_images" / f"{series}.mha") for series in cases.SeriesInstanceUID]
    fields = {view: luna_spec()["views"][view]["field_mm"] for view in ["f32", "f40"]}
    windows = defaultdict(list)
    for name in sorted(cases.batch.unique()):
        with h5py.File(inputs / name, "r") as handle:
            group = handle["w20"]
            for scan_name, start, official in zip(group["scan"][:].astype(str), group["start"][:], group["official"][:]):
                if official:
                    windows[scan_name].append((scan_name, tuple(int(value) for value in start)))
    output = Path(storage["preprocessed"]) / spec["campaign"]
    output.mkdir(parents=True, exist_ok=True)
    groups = [(scan, frame[["identifier", "mask"]].to_dict("records"), windows[Path(scan).stem]) for scan, frame in cases.groupby("scan", sort=True)]
    assert all(len(items) == 1 for _, _, items in groups)
    records, window_values, started = {}, {}, time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        for index, (record_part, window_part) in enumerate(pool.map(scan_distances, [g[0] for g in groups], [g[1] for g in groups], [g[2] for g in groups], [fields] * len(groups), chunksize=8)):
            records.update(record_part)
            window_values.update(window_part)
            if (index + 1) % 250 == 0 or index + 1 == len(groups):
                print(f"distances: {index + 1}/{len(groups)} scans, {(time.monotonic() - started) / 60:.1f} min", flush=True)
    identifiers = sorted(records)
    scans = sorted(window_values)
    with h5py.File(output / "distances.tmp", "w") as handle:
        handle.attrs.update({"code_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG_PATH), "created_at": timestamp()})
        handle.create_dataset("records/identifier", data=np.array(identifiers, dtype="S"))
        for view in fields:
            handle.create_dataset(f"records/{view}", data=np.stack([records[identifier][view] for identifier in identifiers]))
        handle.create_dataset("windows/scan", data=np.array(scans, dtype="S"))
        handle.create_dataset("windows/distance", data=np.stack([window_values[name] for name in scans]), compression="gzip", compression_opts=4)
    (output / "distances.tmp").replace(output / "distances.h5")
    inside = {view: float((np.stack([records[i][view] for i in identifiers]) < 0).mean()) for view in fields}
    write_json(output / "distances.json", {"records": len(identifiers), "windows": len(scans), "inside_token_fraction": inside, "sha256": sha256(output / "distances.h5"), "created_at": timestamp()})
    print("distances written", len(identifiers), "records", len(scans), "windows", inside, flush=True)


def text_embeddings():
    storage, _, _ = context()
    spec = configuration()
    with np.load(Path(storage["runs"]) / "20261018_vlm_text_check" / "embeddings.npz") as saved:
        values = {key: saved[key] for key in ["phrases", "present", "absent"]}
    output = Path(storage["preprocessed"]) / spec["campaign"]
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "text_embeddings.npz", **values)
    print("text embeddings", values["present"].shape, flush=True)


# Remote: random-weight features and final-layer whole-CT tokens.

def random_backbone():
    from .models import published

    manifest = published("flexict")
    sys.path.insert(0, str(Path(manifest["code_directory"])))
    from flexi_ct.flexi_ct_2d import _BACKBONE_KWARGS
    from flexi_ct.models import flexi_ct_backbone_base

    torch.manual_seed(configuration()["random_encoder"]["initialization_seed"])
    backbone = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
    backbone.init_weights()
    return backbone.eval().requires_grad_(False)


def extract(part, parts):
    from .flexict_luna25 import encode, field_input, window_input
    from .models import load_flexict

    spec = configuration()
    storage, _, _ = context()
    inputs = Path(storage["preprocessed"]) / "inputs"
    output = campaign_root() / "cases"
    output.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    trained, _ = load_flexict("ct_3d_vlm.pth")
    random = random_backbone()
    trained.cuda()
    random.cuda()
    state = b"".join(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() for value in random.state_dict().values())
    write_json(output / "random_encoder.json", {**spec["random_encoder"], "state_sha256": hashlib.sha256(state).hexdigest(), "dtypes": sorted({str(value.dtype) for value in random.state_dict().values()})})
    sites = {site: entry for _, site, entry in site_list()}
    window_spec = sites["w20_b16"]
    with h5py.File(campaign_root() / "distances.h5", "r") as handle:
        window_distance = dict(zip(handle["windows/scan"][:].astype(str), handle["windows/distance"][:]))
    cases = pd.read_csv(inputs / "cases.csv", dtype={"PatientID": str}).set_index("identifier")
    batches = sorted(inputs.glob("batch_*.h5"))[part::parts]
    started = time.monotonic()
    for number, path in enumerate(batches):
        target = output / path.name
        if target.exists():
            continue
        with h5py.File(path, "r") as handle, torch.inference_mode():
            identifiers = handle["identifier"][:].astype(str)
            fields = {name: [] for name in ["f32_b10", "f40_b08"]}
            for index in range(len(identifiers)):
                for site in fields:
                    entry = sites[site]
                    x = field_input(handle[entry["view"]][index])
                    fields[site].append(encode(random, x, [entry["block"]])[0][0].float().cpu().numpy())
            group = handle["w20"]
            official = np.flatnonzero(group["official"][:])
            scans = group["scan"][:].astype(str)
            windows = {"scan": [], "uniform_index": [], "trained_uniform": [], "random_uniform": [], "near_index": [], "near_distance": [], "trained_near": [], "random_near": [], "near_count": []}
            full = {"scan": [], "tokens": [], "cls": []}
            for window in official:
                scan = scans[window]
                x = window_input(group["windows"][window], float(group["mean"][window]), float(group["std"][window]))
                (patch, cls), = encode(trained, x, [16])
                (random_patch, _), = encode(random, x, [16])
                patch, random_patch = patch.float().cpu().numpy(), random_patch.float().cpu().numpy()
                uniform = np.sort(np.random.default_rng(stable_value(f"{spec['sampling']['seed']}:w20:{scan}")).choice(len(patch), window_spec["stored_uniform_tokens"], replace=False))
                near = np.flatnonzero(window_distance[scan] <= window_spec["stored_near_mm"])
                windows["scan"].append(scan)
                windows["uniform_index"].append(uniform)
                windows["trained_uniform"].append(patch[uniform])
                windows["random_uniform"].append(random_patch[uniform])
                windows["near_index"].append(near)
                windows["near_distance"].append(window_distance[scan][near])
                windows["trained_near"].append(patch[near])
                windows["random_near"].append(random_patch[near])
                windows["near_count"].append(len(near))
                if cases.loc[cases.SeriesInstanceUID == scan, "split"].iloc[0] == "development":
                    full["scan"].append(scan)
                    full["tokens"].append(patch)
                    full["cls"].append(cls.float().cpu().numpy())
        with h5py.File(target.with_suffix(".tmp"), "w") as handle:
            handle.attrs["source_sha256"] = sha256(path)
            handle.create_dataset("records/identifier", data=identifiers.astype("S"))
            for site, values in fields.items():
                handle.create_dataset(f"records/random_{site}", data=np.stack(values))
            handle.create_dataset("windows/scan", data=np.array(windows["scan"], dtype="S"))
            handle.create_dataset("windows/near_offsets", data=np.r_[0, np.cumsum(windows["near_count"])].astype(np.int64))
            for key in ["uniform_index", "trained_uniform", "random_uniform"]:
                handle.create_dataset(f"windows/{key}", data=np.stack(windows[key]))
            for key in ["near_index", "near_distance", "trained_near", "random_near"]:
                handle.create_dataset(f"windows/{key}", data=np.concatenate(windows[key]))
            if full["scan"]:
                handle.create_dataset("full/scan", data=np.array(full["scan"], dtype="S"))
                handle.create_dataset("full/tokens", data=np.stack(full["tokens"]), chunks=(1, 8000, 864))
                handle.create_dataset("full/cls", data=np.stack(full["cls"]))
        target.with_suffix(".tmp").replace(target)
        print(f"extract part {part}: {path.name} ({number + 1}/{len(batches)}), {len(official)} windows, {(time.monotonic() - started) / 60:.1f} min", flush=True)


# Remote: records, matrices and nodule pools.

def feature_directory(model):
    return Path(read_json(suite_runs() / f"{model}_features.json")["directory"])


def site_records(model, site):
    # one entry per record: identifier (or scan), PatientID, split, trained tokens, random tokens, token distances
    spec = configuration()["models"][model][site]
    inputs = Path(context()[0]["preprocessed"]) / "inputs"
    cases = pd.read_csv(inputs / "cases.csv", dtype={"PatientID": str})
    cases = cases[cases.split != "test"]
    sources = {item["name"]: item["sha256"] for item in read_json(inputs / "inputs.json")["batches"]}
    extracted = sorted((campaign_root() / "cases").glob("batch_*.h5"))
    with h5py.File(campaign_root() / "distances.h5", "r") as handle:
        record_distance = dict(zip(handle["records/identifier"][:].astype(str), handle[f"records/{spec['view']}"][:])) if spec["kind"] == "field_tokens" else None
    if spec["kind"] == "field_tokens":
        lookup = cases.set_index("identifier")
        for path in sorted(feature_directory(model).glob("batch_*.h5")):
            with h5py.File(path, "r") as trained, h5py.File(campaign_root() / "cases" / path.name, "r") as random:
                identifiers = trained["identifier"][:].astype(str)
                assert np.array_equal(identifiers, random["records/identifier"][:].astype(str)) and random.attrs["source_sha256"] == sources[path.name] == trained.attrs["source_sha256"], path.name
                trained_values = trained[f"{spec['view']}/block_{spec['block']:02d}"][:]
                random_values = random[f"records/random_{site}"][:]
            for index, identifier in enumerate(identifiers):
                row = lookup.loc[identifier]
                yield {"record": identifier, "PatientID": row.PatientID, "split": row.split, "trained": trained_values[index], "random": random_values[index], "distance": record_distance[identifier]}
        return
    patients = cases.drop_duplicates("SeriesInstanceUID").set_index("SeriesInstanceUID")
    for path in extracted:
        with h5py.File(path, "r") as handle:
            assert handle.attrs["source_sha256"] == sources[path.name], path.name
            group = handle["windows"]
            offsets = group["near_offsets"][:]
            with h5py.File(campaign_root() / "distances.h5", "r") as distance_handle:
                window_distance = dict(zip(distance_handle["windows/scan"][:].astype(str), range(len(distance_handle["windows/scan"]))))
                distance_values = distance_handle["windows/distance"]
                for index, scan in enumerate(group["scan"][:].astype(str)):
                    uniform = group["uniform_index"][index]
                    near = slice(offsets[index], offsets[index + 1])
                    row = patients.loc[scan]
                    yield {"record": scan, "PatientID": row.PatientID, "split": row.split, "trained": group["trained_uniform"][index], "random": group["random_uniform"][index],
                           "distance": distance_values[window_distance[scan]][uniform], "near_trained": group["trained_near"][near], "near_random": group["random_near"][near], "near_distance": group["near_distance"][near]}


def write_matrix(destination, model, site, encoder, splits, extra):
    channels = configuration()["models"][model][site]["channels"]
    for split, (values, offsets) in splits.items():
        np.save(destination / f"{split}.npy", values)
        np.save(destination / f"{split}_offsets.npy", offsets)
    train, offsets = splits["train"]
    means = np.stack([train[start:end].astype(np.float64).mean(0) for start, end in zip(offsets[:-1], offsets[1:])])
    squares = np.stack([np.square(train[start:end].astype(np.float64)).mean(0) for start, end in zip(offsets[:-1], offsets[1:])])
    mean, square = means.mean(0), squares.mean(0)
    variance = np.maximum(square - mean ** 2, 0)
    scale = float(np.sqrt(variance.mean()))
    assert scale > 0
    np.savez(destination / "normalization.npz", mean=mean.astype(np.float32), scale=np.float32(scale), variance=variance)
    metadata = {"model": model, "site": site, "encoder": encoder, "channels": channels, "kind": configuration()["models"][model][site]["kind"], "sampling": configuration()["sampling"],
                "normalization": "equal training-patient channel mean and global RMS", "splits": {split: {"patients": len(offsets) - 1, "tokens": int(offsets[-1])} for split, (_, offsets) in splits.items()},
                "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(destination.glob("*.np*"))}, "test_used": False, **extra}
    metadata["identity"] = digest({key: value for key, value in metadata.items() if key != "files"} | {"files": sorted((name, item["sha256"]) for name, item in metadata["files"].items())})
    metadata["created_at"] = timestamp()
    write_json(destination / "matrix.json", metadata)
    print("MATRIX", model, site, encoder, metadata["splits"], flush=True)
    return metadata


def write_pool(destination, matrix, bands):
    destination.mkdir(parents=True, exist_ok=True)
    files, counts = {}, {}
    for name, (values, offsets) in bands.items():
        np.save(destination / f"band_{name}.npy", values)
        np.save(destination / f"band_{name}_offsets.npy", offsets)
        for path in [destination / f"band_{name}.npy", destination / f"band_{name}_offsets.npy"]:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        counts[name] = {"tokens": int(len(values)), "patients": int(len(offsets) - 1)}
    identity = hashlib.sha256(("".join(name + info["sha256"] for name, info in sorted(files.items())) + matrix["identity"]).encode()).hexdigest()
    write_json(destination / "pool.json", {"bands": [{"name": name, "low_mm": low, "high_mm": high} for name, low, high in BANDS], "counts": counts, "files": files, "matrix_identity": matrix["identity"],
                                           "identity": identity, "rule": "every training token whose signed distance to the nodule surface lies in the band; one segment per training patient with at least one token", "created_at": timestamp()})
    print("POOL", destination, counts, flush=True)


def matrices():
    spec = configuration()
    root = campaign_root()
    for model, site, entry in site_list():
        if all((root / directory / model / site / "matrix.json").exists() for directory in ["matrices", "matrices_random"]):
            continue
        by_patient = defaultdict(list)
        for record in site_records(model, site):
            by_patient[(record["split"], record["PatientID"])].append(record)
        member_rows, splits, bands = [], {encoder: {} for encoder in ["trained", "random"]}, {encoder: {name: ([], []) for name, _, _ in BANDS} for encoder in ["trained", "random"]}
        for split in ["train", "development"]:
            budget = spec["sampling"][f"{split}_tokens_per_patient"]
            parts = {encoder: [] for encoder in splits}
            lengths = []
            for (_, patient), records in sorted((key, value) for key, value in by_patient.items() if key[0] == split):
                records = sorted(records, key=lambda item: item["record"])
                member_rows += [{"record": item["record"], "PatientID": patient, "split": split} for item in records]
                stacked = {encoder: np.concatenate([item[encoder] for item in records]) for encoder in splits}
                chosen = np.random.default_rng(stable_value(f"{spec['sampling']['seed']}:{model}:{site}:{patient}")).choice(len(stacked["trained"]), size=min(budget, len(stacked["trained"])), replace=False)
                for encoder in splits:
                    parts[encoder].append(stacked[encoder][chosen])
                lengths.append(len(chosen))
                if split == "train":
                    distance = np.concatenate([item["near_distance"] if "near_distance" in item else item["distance"] for item in records])
                    for encoder in splits:
                        source = np.concatenate([item[f"near_{encoder}"] if "near_distance" in item else item[encoder] for item in records])
                        for name, low, high in BANDS:
                            selected = (distance >= low) & (distance < high)
                            if selected.any():
                                bands[encoder][name][0].append(source[selected])
                                bands[encoder][name][1].append(int(selected.sum()))
            offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
            for encoder in splits:
                splits[encoder][split] = (np.concatenate(parts[encoder]).astype(np.float32), offsets)
        paired = None
        for encoder, directory in [("trained", "matrices"), ("random", "matrices_random")]:
            destination = root / directory / model / site
            destination.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(member_rows).to_csv(destination / "members.csv", index=False)
            matrix = write_matrix(destination, model, site, encoder, splits[encoder], {"paired_with": paired} if paired else {})
            paired = paired or matrix["identity"]
            write_pool(root / "nodule_pools" / encoder / model / site, matrix, {name: (np.concatenate(values).astype(np.float32), np.r_[0, np.cumsum(counts)].astype(np.int64)) for name, (values, counts) in bands[encoder].items()})
        del by_patient, splits, bands
        gc.collect()


# Remote: jobs and workers.

def run_id(encoder, model, site, expansion, k, seed, variant=None):
    value = f"{configuration()['campaign']}_{encoder}_{model}_{site}_e{expansion}_k{k}_s{seed}"
    return f"{value}_{variant}" if variant else value


# ordered screens: phase -> (configuration key, variant prefix, stage)
ORDERED_PHASES = {"head_screen": ("head_read", "read", "screen"), "head_seeds": ("head_read", "read", "seeds"),
                  "finetune_screen": ("head_read_finetune", "ft", "screen"), "finetune_seeds": ("head_read_finetune", "ft", "seeds")}


def head_read_training(settings, prefix, model, site, k, weight, seed, head_run):
    terms = {"pooled_loss": {"weight": weight}, "head_space": {"weight": weight, "head_run": head_run}}
    if prefix == "ft":
        # fine-tuning starts from the token-loss run of the same seed and k
        terms.update({"steps": settings["steps"], "init_from": run_id("trained", model, site, configuration()["grid"]["expansion"][0], k, seed)})
    return terms


def write_jobs(phase, choices=None):
    # grid: trained uniform dictionaries over k; head_screen / finetune_screen: reference seed of every head-read (from scratch / fine-tuned) configuration
    # at sites without a qualifying configuration; head_seeds / finetune_seeds: the other seeds, configuration by configuration;
    # variants: at each site's chosen k the variant runs and the random-weight controls of the uniform and nodule-pool dictionaries;
    # production: 8 trained and 8 random seeds of a chosen 192,000-step configuration
    spec = configuration()
    root = campaign_root()
    (root / "jobs").mkdir(parents=True, exist_ok=True)
    grid = spec["grid"]
    variants = {item["name"]: item for item in spec["variants"]}
    jobs = []
    for model, site, _ in site_list():
        base = {"model": model, "site": site, "expansion": grid["expansion"][0]}
        if phase == "grid":
            jobs += [{**base, "seed": seed, "k": k, "encoder": "trained"} for k in grid["k"] for seed in grid["seeds"]]
            continue
        if phase == "extension":
            if f"{model}:{site}" in choices["sites"]:
                jobs += [{**base, "seed": seed, "k": k, "encoder": "trained"} for k in spec["grid_extension"]["k"] for seed in grid["seeds"]]
            continue
        if phase == "downstream":
            if f"{model}:{site}" in choices["sites"]:
                jobs += [{**base, "seed": seed, "k": k, "encoder": "trained", "variant": f"e2e_w{weight}", "training": {"output_loss": {"weight": weight}}}
                         for k in spec["downstream"]["k"] for weight in spec["downstream"]["weights"] for seed in grid["seeds"]]
            continue
        if phase in ORDERED_PHASES:
            settings_key, prefix, stage = ORDERED_PHASES[phase]
            settings = spec[settings_key]
            if choices[f"{model}:{site}"]["k"] is None:
                head_run = str(deployed_head(model, site)[1])
                make_job = lambda k, weight, seed: {**base, "seed": seed, "k": k, "encoder": "trained", "variant": f"{prefix}_w{weight}", "training": head_read_training(settings, prefix, model, site, k, weight, seed, head_run)}
                if stage == "screen":
                    jobs += [make_job(k, weight, grid["reference_seed"]) for k in settings["k"] for weight in settings["weights"]]
                    continue
                # configurations in order (k, then weight); a configuration whose reference screen fails or whose gate can no longer be met is ruled out;
                # stop at the first unevaluated screen, at the first open configuration (writing its other seeds when missing), or at the first passing one
                for k, weight in [(k, weight) for k in settings["k"] for weight in settings["weights"]]:
                    group = f"trained_{model}_{site}_e{base['expansion']}_k{k}_{prefix}_w{weight}"
                    screen_path = root / "jobs" / f"{run_id('trained', model, site, base['expansion'], k, grid['reference_seed'], f'{prefix}_w{weight}')}.json"
                    if not screen_path.exists() or not (root / "evaluation" / f"{group}.json").exists():
                        break
                    if not reference_passes(group) or gate_status(group) == "failed":
                        continue
                    if gate_status(group) == "open":
                        jobs += [make_job(k, weight, seed) for seed in grid["seeds"] if seed != grid["reference_seed"]]
                    break
            continue
        choice = choices[f"{model}:{site}"]
        if choice["k"] is None:
            continue
        # sites chosen through an output-aware objective keep its terms in every trained variant; random-weight controls keep the token loss only;
        # a fine-tuning objective first trains the token-loss run of the variant, then fine-tunes it from that run
        objective = choice.get("objective")
        prefix = f"{objective['name']}_" if objective else ""

        def trained_jobs(name, training, seeds, sources=None):
            # sources: seed -> variant of an existing token-loss run with identical training, used as the fine-tuning start instead of a new run
            sources = sources or {}
            if objective is None:
                return [{**base, "seed": seed, "k": choice["k"], "encoder": "trained", "variant": name, "training": training} for seed in seeds]
            if "init" not in objective:
                return [{**base, "seed": seed, "k": choice["k"], "encoder": "trained", "variant": prefix + name, "training": {**training, **objective["training"]}} for seed in seeds]
            return ([{**base, "seed": seed, "k": choice["k"], "encoder": "trained", "variant": name, "training": training} for seed in seeds if seed not in sources]
                    + [{**base, "seed": seed, "k": choice["k"], "encoder": "trained", "variant": prefix + name,
                        "training": {**training, **objective["training"], "init_from": run_id("trained", model, site, base["expansion"], choice["k"], seed, sources.get(seed, name))}} for seed in seeds])

        if phase == "variants":
            for name, variant in variants.items():
                jobs += trained_jobs(name, variant["training"], grid["seeds"] if variant["seeds"] == "grid" else variant["seeds"])
            jobs += [{**base, "seed": seed, "k": choice["k"], "encoder": "random"} for seed in grid["seeds"]]
            jobs += [{**base, "seed": seed, "k": choice["k"], "encoder": "random", "variant": "nodule75", "training": variants["nodule75"]["training"]} for seed in grid["seeds"]]
        elif choice.get("steps", spec["training"]["steps"]) != spec["training"]["steps"]:
            training = {"steps": choice["steps"], **({"nodule_pool": variants["nodule75"]["training"]["nodule_pool"]} if choice["pool"] else {})}
            name = f"{'nodule75' if choice['pool'] else 'uniform'}_{choice['steps'] // 1000}k"
            # tokens16x token-loss runs have exactly this training for their seeds, so fine-tuning starts from them
            long = variants["tokens16x"]
            sources = {seed: "tokens16x" for seed in long["seeds"]} if objective and "init" in objective and training == long["training"] else {}
            jobs += trained_jobs(name, training, grid["seeds"], sources)
            jobs += [{**base, "seed": seed, "k": choice["k"], "encoder": "random", "variant": name, "training": training} for seed in grid["seeds"]]
    written = 0
    for job in jobs:
        job["run_id"] = run_id(job["encoder"], job["model"], job["site"], job["expansion"], job["k"], job["seed"], job.get("variant"))
        job["matrix_directory"] = "matrices" if job["encoder"] == "trained" else "matrices_random"
        path = root / "jobs" / f"{job['run_id']}.json"
        if path.exists():
            assert read_json(path) == job, job["run_id"]
            continue
        write_json(path, job)
        written += 1
    print("JOBS", phase, written, "new of", len(jobs), flush=True)


def other_processes(gpu):
    # GPU processes of other users and of the same account's other projects (sae-consistency runs as the same user);
    # every process of this project, spawned job processes included, runs this project's interpreter, whose path is in its command line
    uuid = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.strip()
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    others = []
    for row in csv.reader(io.StringIO(rows)):
        if not row or row[0].strip() != uuid:
            continue
        command = subprocess.run(["ps", "-o", "args=", "-p", row[1].strip()], capture_output=True, text=True).stdout.strip()
        # a process that exited between the two queries has no command line
        if command and sys.executable not in command:
            others.append({"pid": int(row[1]), "command": command[:200], "memory_mib": int(row[2])})
    return others


def yield_control(poll_seconds=30, free_minutes=10):
    # pauses the workers of a GPU while another project uses it, and resumes them after the GPU has been free for free_minutes;
    # when other projects use every GPU, the campaign keeps one (the one where they hold least memory, kept while all stay busy), so both projects run;
    # the training loops read controls.json every poll_seconds of the configuration and checkpoint when paused
    spec = configuration()
    root = campaign_root()
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True, check=True).stdout.split()
    # a GPU that is free when the controller starts counts as free for the whole waiting time
    free_since, previous, kept = {gpu: time.monotonic() - free_minutes * 60 for gpu in gpus}, None, None
    while True:
        others = {gpu: other_processes(gpu) for gpu in gpus}
        memory = {gpu: sum(item["memory_mib"] for item in others[gpu]) for gpu in gpus}
        busy = {gpu: memory[gpu] > spec["control"]["foreign_memory_mib"] for gpu in gpus}
        previous_kept = kept
        kept = (kept if kept is not None else min(gpus, key=lambda gpu: (memory[gpu], gpu))) if all(busy.values()) else None
        workers = {}
        for gpu in gpus:
            if busy[gpu]:
                free_since[gpu] = None
            elif gpu == previous_kept:
                # the GPU the campaign kept while all were busy needs no waiting once the other project leaves it
                free_since[gpu] = time.monotonic() - free_minutes * 60
            elif free_since[gpu] is None:
                free_since[gpu] = time.monotonic()
            paused = gpu != kept and (free_since[gpu] is None or time.monotonic() - free_since[gpu] < free_minutes * 60)
            workers.update({f"w{gpu}{index}": {"paused": paused, "others": [item["command"][:120] for item in others[gpu]]} for index in range(spec["control"]["workers_per_gpu"])})
        state = {name: value["paused"] for name, value in workers.items()}
        write_json(root / "controls.json", {"workers": workers, "updated_at": timestamp()})
        if state != previous:
            print("CONTROL", timestamp(), state, flush=True)
            previous = state
        time.sleep(poll_seconds)


def worker(gpu, name):
    os.environ["SCLVMI_SAE_CAMPAIGN_CONFIG"] = str(CONFIG_PATH.relative_to(ROOT))
    from .sae_campaign import pause_requested

    spec = configuration()
    root = campaign_root()
    (root / "workers").mkdir(parents=True, exist_ok=True)
    # GPU lending is decided by the yield stage through controls.json, which must be running alongside the workers
    assert (root / "controls.json").exists(), "start the yield stage before the workers"
    while True:
        if pause_requested(root, name):
            write_json(root / "workers" / f"{name}.json", {"state": "paused", "updated_at": timestamp()})
            time.sleep(15)
            continue
        chosen = None
        # reference-seed jobs first, then fine-tuning jobs: both are short or decide which later jobs are written
        queue = sorted(((read_json(path), path) for path in (root / "jobs").glob("*.json")),
                       key=lambda item: (item[0]["seed"] != spec["grid"]["reference_seed"], "init_from" not in item[0].get("training", {}), item[1].name))
        for job, path in queue:
            run = root / "runs" / job["run_id"]
            if (run / "result.json").exists() or not (root / job["matrix_directory"] / job["model"] / job["site"] / "matrix.json").exists():
                continue
            # a fine-tuning job waits for its token-loss source run
            source = job.get("training", {}).get("init_from")
            if source and not (root / "runs" / source / "result.json").exists():
                continue
            run.mkdir(parents=True, exist_ok=True)
            claim = FileLock(str(run / "assignment.lock"), timeout=0)
            try:
                claim.acquire()
            except Timeout:
                continue
            if (run / "result.json").exists():
                claim.release()
                continue
            chosen = (job, claim)
            break
        if chosen is None:
            write_json(root / "workers" / f"{name}.json", {"state": "finished", "updated_at": timestamp()})
            print("WORKER DONE", name, flush=True)
            return
        job, claim = chosen
        write_json(root / "workers" / f"{name}.json", {"state": "running", "run_id": job["run_id"], "gpu": gpu, "updated_at": timestamp()})
        # each job trains in its own process, so a finished or paused job leaves no GPU memory behind; the worker itself never initializes CUDA
        process = multiprocessing.get_context("spawn").Process(target=importlib.import_module(".flexict_sae", __package__).train_job, args=(job, name))
        process.start()
        process.join()
        claim.release()
        assert process.exitcode == 0, f"{job['run_id']} stopped with exit code {process.exitcode}"


def train_job(job, name):
    from .sae_campaign_train import fit_job

    torch.backends.cuda.matmul.allow_tf32 = False
    training = job.get("training", {})
    result = (fit_downstream if training.get("output_loss") or training.get("pooled_loss") else fit_job)(campaign_root(), job, name)
    print("RESULT", job["run_id"], result["state"], round(result["development"]["fvu"], 4) if "development" in result else "", flush=True)


# Remote: the error the deployed head reads (head-space term of the head-read objective).

def head_projection(head):
    # the max head reads A (x - mean) / scale + b for every token before the maximum over tokens
    return head.network.projector[0].weight.detach() / head.scale


def head_maxima(head, tokens):
    # (bags, tokens, channels) -> the 128 per-bag maxima that the head's classifier reads
    return head.network.projector((tokens - head.mean) / head.scale).amax(1)


@torch.inference_mode()
def projected_variance(values, projection, batch=65536):
    total, square = torch.zeros(len(projection), dtype=torch.float64, device="cuda"), 0.0
    for start in range(0, len(values), batch):
        chunk = values[start:start + batch]
        chunk = torch.from_numpy(np.array(chunk, dtype=np.float32)) if isinstance(chunk, np.ndarray) else chunk
        projected = (chunk.cuda() @ projection.T).double()
        total += projected.sum(0)
        square += float(projected.square().sum())
    return square / len(values) - float((total / len(values)).square().sum())


def head_space_loss(root, job):
    spec = job["training"]["head_space"]
    head, head_run = deployed_head(job["model"], job["site"])
    assert str(head_run) == spec["head_run"], head_run
    matrix = root / job["matrix_directory"] / job["model"] / job["site"]
    with np.load(matrix / "normalization.npz") as normalization:
        scale = float(normalization["scale"])
    projection = head_projection(head)
    variance = projected_variance(np.load(matrix / "train.npy", mmap_mode="r"), projection)
    # batch and reconstruction are in the dictionary's normalized space; their difference times scale is the token error
    return lambda batch, rebuilt: spec["weight"] * (((rebuilt - batch) * scale) @ projection.T).square().sum(-1).mean() / variance


# Remote: token loss plus the deployed head's output (end-to-end style).

def head_bags(model, site, split):
    spec = configuration()["models"][model][site]
    cases = pd.read_csv(Path(context()[0]["preprocessed"]) / "inputs" / "cases.csv", dtype={"PatientID": str})
    chosen = cases[cases.split == split].sort_values("identifier")
    dataset = f"{spec['view']}/block_{spec['block']:02d}"
    values = {}
    for path in sorted(feature_directory(model).glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            identifiers = handle["identifier"][:].astype(str)
            keep = np.flatnonzero(np.isin(identifiers, chosen.identifier))
            if len(keep):
                data = handle[dataset][:]
                values.update({identifiers[index]: data[index] for index in keep})
    return chosen.reset_index(drop=True), np.stack([values[identifier] for identifier in chosen.identifier])


def batch_topk_reconstruct(sae, x):
    # the BatchTopK training forward without updating the inactive-feature counters
    acts = torch.nn.functional.relu((x - sae.b_dec) @ sae.W_enc)
    selected = torch.topk(acts.flatten(), sae.cfg["top_k"] * len(x), sorted=False)
    codes = torch.zeros_like(acts.flatten()).scatter(0, selected.indices, selected.values).reshape(acts.shape)
    return codes @ sae.W_dec + sae.b_dec


def threshold_reconstruct(sae, x, threshold):
    # inference-time reconstruction with the calibrated threshold, as SpatialDictionary uses
    acts = torch.nn.functional.relu((x - sae.b_dec) @ sae.W_enc)
    return (acts * (acts > threshold)) @ sae.W_dec + sae.b_dec


def bernoulli_kl(original, recovered):
    p, q = original.clamp(1e-6, 1 - 1e-6), recovered.clamp(1e-6, 1 - 1e-6)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()


def fit_downstream(root, job, worker):
    import math

    from .sae import SpatialDictionary, export_dictionary, load_matrix
    from .sae_campaign import ensure_identity, pause_requested, training_identity
    from .sae_campaign_train import calibrate_patient_balanced, evaluate_weighted, feature_checks, make_sae, patient_indices, save_checkpoint

    config = configuration()
    destination = root / "runs" / job["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    matrix = root / job["matrix_directory"] / job["model"] / job["site"]
    matrix_info = read_json(matrix / "matrix.json")
    head, head_run = deployed_head(job["model"], job["site"])
    head.requires_grad_(False)
    # output_loss: Bernoulli KL of the head's probability (downstream); pooled_loss with head_space: the head-read objective (head_read)
    objective = "pooled_loss" if "pooled_loss" in job.get("training", {}) else "output_loss"
    request = {**training_identity(config, job, matrix_info), "head_run": str(head_run), "downstream": config["head_read"]["loss"] if objective == "pooled_loss" else config["downstream"]["loss"]}
    pool_spec = request["training"].get("nodule_pool")
    if pool_spec:
        pool_path = root / "nodule_pools" / job["encoder"] / job["model"] / job["site"]
        pool_info = read_json(pool_path / "pool.json")
        assert pool_info["matrix_identity"] == matrix_info["identity"] and [band["name"] for band in pool_info["bands"]] == pool_spec["bands"], pool_path
        request["nodule_pool_identity"] = pool_info["identity"]
    # fine-tuning starts from a token-loss dictionary of the same matrix, k and expansion
    source = request["training"].get("init_from")
    if source:
        source_result = read_json(root / "runs" / source / "result.json")
        assert all(source_result["request"][name] == request[name] for name in ["matrix_identity", "k", "dict_size", "model", "site"]), source
        request["init_identity"] = {"run_id": source, "dictionary_sha256": source_result["dictionary_sha256"]}
    ensure_identity(destination / "request.json", request)
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    if pause_requested(root, worker):
        return {"state": "paused"}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(matrix / "normalization.npz") as normalization:
        mean = torch.tensor(normalization["mean"], device="cuda")
        scale = float(normalization["scale"])
    train = load_matrix(matrix / "train.npy", mean, scale)
    development = load_matrix(matrix / "development.npy", mean, scale)
    train_offsets = torch.tensor(np.load(matrix / "train_offsets.npy"), device="cuda", dtype=torch.int64)
    dev_offsets = torch.tensor(np.load(matrix / "development_offsets.npy"), device="cuda", dtype=torch.int64)
    pools =[(load_matrix(pool_path / f"band_{name}.npy", mean, scale), torch.tensor(np.load(pool_path / f"band_{name}_offsets.npy"), device="cuda", dtype=torch.int64)) for name in pool_spec["bands"]] if pool_spec else []
    frame, bags = head_bags(job["model"], job["site"], "train")
    bags = torch.from_numpy(bags).cuda()
    with torch.no_grad():
        if objective == "pooled_loss":
            target = torch.cat([head_maxima(head, bags[start:start + 64]) for start in range(0, len(bags), 64)])
            spread = float((target - target.mean(0)).square().sum(-1).mean())
            # development bags are only reported along training; the final checkpoint is kept regardless
            _, watch = development_tokens(job["model"], job["site"])
            watch = watch.cuda()
            watch_probability = torch.cat([head(watch[start:start + 64]).sigmoid() for start in range(0, len(watch), 64)])
            watch_maxima = torch.cat([head_maxima(head, watch[start:start + 64]) for start in range(0, len(watch), 64)])
            watch_spread = float((watch_maxima - watch_maxima.mean(0)).square().sum(-1).mean())
        else:
            target = torch.cat([head(bags[start:start + 64]).sigmoid() for start in range(0, len(bags), 64)])
    patients = frame.groupby("PatientID", sort=True).indices
    record_offsets = torch.tensor(np.r_[0, np.cumsum([len(rows) for rows in patients.values()])], device="cuda", dtype=torch.int64)
    record_order = torch.tensor(np.concatenate(list(patients.values())), device="cuda", dtype=torch.int64)
    sae = make_sae(request, config)
    if source:
        assert sha256(root / "runs" / source / "dictionary.pt") == request["init_identity"]["dictionary_sha256"], source
        sae.load_state_dict(torch.load(root / "runs" / source / "dictionary.pt", map_location="cuda", weights_only=True)["state_dict"], strict=True)
    training = request["training"]
    weight, bag_records = training[objective]["weight"], config["downstream"]["bag_records"]
    term_name = "train_pooled_fvu" if objective == "pooled_loss" else "train_kl"
    token_term = head_space_loss(root, job) if training.get("head_space") else None
    optimizer = torch.optim.Adam(sae.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]))
    generator = torch.Generator(device="cuda").manual_seed(job["seed"] + 12345)
    history, start_step, previous_elapsed = [], 0, 0.0
    if (destination / "resume.pt").exists():
        saved = torch.load(destination / "resume.pt", map_location="cuda", weights_only=False)
        assert saved["identity"] == request
        sae.load_state_dict(saved["state_dict"], strict=True)
        sae.num_batches_not_active.copy_(saved["inactive"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        history, start_step, previous_elapsed = saved["history"], saved["step"], saved["elapsed_seconds"]
    total, batch_size = training["steps"], min(training["batch_size"], len(train))
    pooled = round(batch_size * pool_spec["fraction"]) if pool_spec else 0
    shares = [pooled // len(pools) + int(index < pooled % len(pools)) for index in range(len(pools))]
    started, last_checkpoint, last_control, stopped = time.monotonic(), time.monotonic(), time.monotonic(), False
    for step in range(start_step + 1, total + 1):
        factor = min(1.0, step / min(training["warmup_steps"], max(1, total // 10)))
        if step > int(total * training["decay_start_fraction"]):
            factor *= 0.1 + 0.9 * (total - step) / max(1, total - int(total * training["decay_start_fraction"]))
        for group in optimizer.param_groups:
            group["lr"] = training["learning_rate"] * factor
        # the nodule-pool variant draws a fixed share of each token batch evenly from the distance bands, as in fit_job
        batch = torch.cat([train[patient_indices(train_offsets, batch_size - pooled, generator)]] + [values[patient_indices(offsets, share, generator)] for (values, offsets), share in zip(pools, shares)])
        output = sae(batch)
        records = record_order[patient_indices(record_offsets, bag_records, generator)]
        tokens = (bags[records] - mean) / scale
        rebuilt = batch_topk_reconstruct(sae, tokens.reshape(-1, tokens.shape[-1])).reshape(tokens.shape) * scale + mean
        if objective == "pooled_loss":
            term = (head_maxima(head, rebuilt) - target[records]).square().sum(-1).mean() / spread
        else:
            term = bernoulli_kl(target[records], head(rebuilt).sigmoid())
        loss = output["loss"] + weight * term + (token_term(batch, output["sae_out"]) if token_term else 0)
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), training["gradient_clip"], error_if_nonfinite=True)
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % training["validate_every"] == 0 or step == total:
            threshold = calibrate_patient_balanced(sae, train, train_offsets, request["inference"]["calibration_train_tokens"], batch_size)
            metrics, _ = evaluate_weighted(sae, development, dev_offsets, threshold, mean, scale, batch_size)
            metrics.update({"step": step, term_name: float(term.detach()), "updated_at": timestamp()})
            if objective == "pooled_loss":
                with torch.no_grad():
                    probability, maxima = [], []
                    for start in range(0, len(watch), 64):
                        part = watch[start:start + 64]
                        rebuilt = threshold_reconstruct(sae, ((part - mean) / scale).reshape(-1, part.shape[-1]), threshold).reshape(part.shape) * scale + mean
                        probability.append(head(rebuilt).sigmoid())
                        maxima.append(head_maxima(head, rebuilt))
                metrics.update({"development_probability_mae": float((torch.cat(probability) - watch_probability).abs().mean()),
                                "development_pooled_fvu": float((torch.cat(maxima) - watch_maxima).square().sum(-1).mean()) / watch_spread})
            history.append(metrics)
            write_json(destination / "history.json", history)
            export_dictionary(sae, request, threshold, mean, scale, destination / "best.pt", step)
            print("VALIDATION", job["run_id"], round(metrics["fvu"], 4), round(metrics[term_name], 5), round(metrics.get("development_probability_mae", math.nan), 4), round(metrics.get("development_pooled_fvu", math.nan), 4), flush=True)
        if time.monotonic() - last_control >= config["control"]["poll_seconds"]:
            stopped = pause_requested(root, worker) and step < total
            last_control = time.monotonic()
        if step % training["checkpoint_every"] == 0 or time.monotonic() - last_checkpoint >= training["checkpoint_seconds"] or step == total or stopped:
            save_checkpoint(destination, sae, optimizer, generator, step, history, math.inf, previous_elapsed + time.monotonic() - started, request)
            last_checkpoint = time.monotonic()
        if step % 2000 == 0 or stopped:
            write_json(destination / "status.json", {"state": "paused" if stopped else "running", "step": step, "total": total, "worker": worker, "updated_at": timestamp()})
        if stopped:
            return {"state": "paused", "step": step}
    best = torch.load(destination / "best.pt", map_location="cuda", weights_only=True)
    sae.load_state_dict(best["state_dict"], strict=True)
    metrics, statistics = evaluate_weighted(sae, development, dev_offsets, best["threshold"], mean, scale, batch_size)
    np.savez(destination / "development_feature_statistics.npz", **statistics)
    export_dictionary(sae, request, best["threshold"], mean, scale, destination / "dictionary.pt", best["step"])
    loaded = SpatialDictionary(destination / "dictionary.pt")
    actual = development[:127] * scale + mean
    # batch independence before the elementwise threshold, where rounding may flip a value lying at the threshold
    torch.testing.assert_close(loaded.preactivations(actual), torch.cat([loaded.preactivations(part) for part in actual.split(17)]), rtol=1e-5, atol=config["inference"]["batch_independence_atol"])
    checks = feature_checks(metrics, request)
    result = {"state": "completed", "run_id": job["run_id"], "request": request, "steps": total, "best_step": best["step"], "development": metrics, "feature_checks": checks,
              "feature_checks_pass": all(checks.values()), "dictionary_sha256": sha256(destination / "dictionary.pt"), "elapsed_seconds": previous_elapsed + time.monotonic() - started,
              "completed_at": timestamp(), "test_used": False}
    write_json(destination / "result.json", result)
    write_json(destination / "status.json", {"state": "completed", "worker": worker, "step": total, "total": total, "updated_at": timestamp()})
    del sae, optimizer, train, development, bags, pools
    gc.collect()
    torch.cuda.empty_cache()
    return result


# Remote: fixed-classifier and text-fidelity checks.

def deployed_head(model, site):
    from .frozen_heads import SpatialHead

    selection = read_json(suite_runs() / "heads" / model / "selection.json")
    if site in selection["runs"]:
        run = Path(selection["runs"][site][0])
    else:
        options = sorted((suite_runs() / "heads" / model / site).glob(f"{model}_max_c*_s2025"))
        run = max(options, key=lambda path: (read_json(path / "result.json")["metrics"]["auroc"], -read_json(path / "result.json")["metrics"]["log_loss"]))
    assert read_json(run / "result.json")["seed"] == 2025, run
    head = SpatialHead(configuration()["models"][model][site]["channels"], "max", np.zeros(864, dtype=np.float32), np.ones(864, dtype=np.float32)).cuda()
    head.load_state_dict(torch.load(run / "best.pt", map_location="cpu", weights_only=True))
    return head.eval(), run


def development_tokens(model, site):
    spec = configuration()["models"][model][site]
    cases = pd.read_csv(Path(context()[0]["preprocessed"]) / "inputs" / "cases.csv", dtype={"PatientID": str})
    development = cases[cases.split == "development"].sort_values("identifier")
    dataset = f"{spec['view']}/block_{spec['block']:02d}"
    values = {}
    for path in sorted(feature_directory(model).glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            identifiers = handle["identifier"][:].astype(str)
            keep = np.flatnonzero(np.isin(identifiers, development.identifier))
            if len(keep):
                data = handle[dataset][:]
                values.update({identifiers[index]: data[index] for index in keep})
    return development, torch.from_numpy(np.stack([values[identifier] for identifier in development.identifier]))


@torch.inference_mode()
def classifier_check(model, site, runs):
    from .frozen_heads import probabilities
    from .head import metrics
    from .sae import SpatialDictionary

    limits = configuration()["acceptance"]
    head, head_run = deployed_head(model, site)
    frame, tokens = development_tokens(model, site)
    original = probabilities(head, tokens)
    base = metrics(frame.label, original)
    projection = head_projection(head)
    variance = projected_variance(tokens.reshape(-1, tokens.shape[-1]), projection)
    maxima = torch.cat([head_maxima(head, part.cuda()) for part in tokens.split(64)])
    spread = float((maxima - maxima.mean(0)).square().sum(-1).mean())
    rows = []
    for run in runs:
        dictionary = SpatialDictionary(campaign_root() / "runs" / run / "dictionary.pt")
        shape = tokens.shape
        rebuilt = torch.cat([dictionary.reconstruct(part.reshape(-1, shape[-1]).cuda()).reshape(-1, *shape[1:]).cpu() for part in tokens.split(64)])
        recovered = probabilities(head, rebuilt)
        score = metrics(frame.label, recovered)
        error = sum(float(((after.reshape(-1, shape[-1]).cuda() - before.reshape(-1, shape[-1]).cuda()) @ projection.T).double().square().sum()) for after, before in zip(rebuilt.split(64), tokens.split(64)))
        recovered_maxima = torch.cat([head_maxima(head, part.cuda()) for part in rebuilt.split(64)])
        row = {"run_id": run, "probability_mae": float(np.abs(recovered - original).mean()), "auroc": score["auroc"], "average_precision": score["average_precision"],
               "auroc_decrease": base["auroc"] - score["auroc"], "ap_decrease": base["average_precision"] - score["average_precision"], "head_space_fvu": error / (shape[0] * shape[1]) / variance,
               "pooled_fvu": float((recovered_maxima - maxima).square().sum(-1).mean()) / spread}
        row["passes"] = bool(row["probability_mae"] <= limits["max_probability_mae"] and row["auroc_decrease"] <= limits["max_auroc_decrease"] and row["ap_decrease"] <= limits["max_ap_decrease"])
        rows.append(row)
    return {"head_run": str(head_run), "original": base, "development_records": len(frame), "tokens_per_record": int(tokens.shape[1]), "runs": rows}


@torch.inference_mode()
def text_check(runs):
    from .models import published
    from .sae import SpatialDictionary

    storage, _, _ = context()
    # the checkpoint stores the projection in bfloat16; the evaluation runs in float32
    projection = torch.load(Path(published("flexict")["directory"]) / "ct_3d_vlm.pth", map_location="cpu", weights_only=True)["model"]["vlm_vision_projection.weight"].float().cuda()
    with np.load(campaign_root() / "text_embeddings.npz") as saved:
        phrases, present, absent = saved["phrases"].astype(str), torch.from_numpy(saved["present"]).cuda(), torch.from_numpy(saved["absent"]).cuda()
    present, absent = torch.nn.functional.normalize(present, dim=-1), torch.nn.functional.normalize(absent, dim=-1)

    def embed(cls, mean):
        return torch.nn.functional.normalize(torch.cat([cls, mean], -1) @ projection.T, dim=-1)

    dictionaries = {run: SpatialDictionary(campaign_root() / "runs" / run / "dictionary.pt") for run in runs}
    original, rebuilt = [], {run: [] for run in runs}
    for path in sorted((campaign_root() / "cases").glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            if "full" not in handle:
                continue
            for index in range(len(handle["full/scan"])):
                tokens = torch.from_numpy(handle["full/tokens"][index]).cuda()
                cls = torch.from_numpy(handle["full/cls"][index]).cuda()
                original.append(embed(cls, tokens.mean(0)))
                for run, dictionary in dictionaries.items():
                    rebuilt[run].append(embed(cls, dictionary.reconstruct(tokens).mean(0)))
    original = torch.stack(original)
    scores = lambda values: (values @ present.T - values @ absent.T).cpu().numpy()
    base = scores(original)
    rows = []
    rule = configuration()["text_fidelity"]
    for run in runs:
        values = torch.stack(rebuilt[run])
        cosine = float((values * original).sum(-1).mean())
        correlation = [spearmanr(base[:, index], scores(values)[:, index]).statistic for index in range(len(phrases))]
        rows.append({"run_id": run, "image_cosine": cosine, "median_score_spearman": float(np.median(correlation)), "minimum_score_spearman": float(np.min(correlation)),
                     "passes": bool(cosine >= 0.99 and np.median(correlation) >= 0.90)})
    return {"windows": len(original), "phrases": phrases.tolist(), "rule": rule["rule"], "runs": rows}


def evaluate(groups):
    root = campaign_root()
    output = root / "evaluation"
    output.mkdir(exist_ok=True)
    for group in groups:
        jobs = sorted([read_json(path) for path in (root / "jobs").glob("*.json") if group_key(read_json(path)) == group], key=lambda job: job["seed"])
        # the deployed heads read trained-encoder tokens, so the checks apply to trained-encoder dictionaries only
        assert jobs[0]["encoder"] == "trained", group
        runs = [job["run_id"] for job in jobs if (root / "runs" / job["run_id"] / "result.json").exists()]
        # a group screened with its reference seed gains seeds later; only runs without a result are evaluated and merged in seed order
        previous = read_json(output / f"{group}.json") if (output / f"{group}.json").exists() else None
        done = {row["run_id"] for row in previous["classifier"]["runs"]} if previous else set()
        new = [run for run in runs if run not in done]
        # rows written before head_space_fvu and pooled_fvu were reported are recomputed, and their gate values must agree
        stale = [row["run_id"] for row in previous["classifier"]["runs"] if "pooled_fvu" not in row] if previous else []
        if not new and not stale:
            continue
        model, site = jobs[0]["model"], jobs[0]["site"]
        result = {"group": group, "classifier": classifier_check(model, site, stale + new), "created_at": timestamp(), "test_used": False}
        if f"{model}:{site}" in configuration()["text_fidelity"]["sites"] and jobs[0]["encoder"] == "trained" and new:
            result["text"] = text_check(new)
        if previous:
            assert previous["classifier"]["head_run"] == result["classifier"]["head_run"] and all(abs(previous["classifier"]["original"][name] - value) < 1e-6 for name, value in result["classifier"]["original"].items()), group
            recomputed = {row["run_id"]: row for row in result["classifier"]["runs"]}
            for row in previous["classifier"]["runs"]:
                if row["run_id"] in stale:
                    assert all(abs(row[name] - recomputed[row["run_id"]][name]) < 1e-6 for name in ["probability_mae", "auroc", "average_precision"]) and row["passes"] == recomputed[row["run_id"]]["passes"], row["run_id"]
            order = {run: index for index, run in enumerate(runs)}
            result["classifier"]["runs"] = sorted([recomputed.get(row["run_id"], row) for row in previous["classifier"]["runs"]] + [recomputed[run] for run in new], key=lambda row: order[row["run_id"]])
            if "text" in previous:
                result["text"] = {**result.get("text", previous["text"]), "runs": sorted(previous["text"]["runs"] + result.get("text", {}).get("runs", []), key=lambda row: order[row["run_id"]])}
            result["merged_from"] = previous["created_at"]
        write_json(output / f"{group}.json", result)
        print("EVALUATE", group, [(row["run_id"][-6:], round(row["probability_mae"], 4), round(row["auroc_decrease"], 4), row["passes"]) for row in result["classifier"]["runs"]],
              [(round(row["image_cosine"], 4), round(row["median_score_spearman"], 3), row["passes"]) for row in result.get("text", {}).get("runs", [])], flush=True)
        torch.cuda.empty_cache()


# Remote: stability and nodule coverage.

def group_key(job):
    key = f"{job['encoder']}_{job['model']}_{job['site']}_e{job['expansion']}_k{job['k']}"
    return f"{key}_{job['variant']}" if "variant" in job else key


def stability(minimum_seeds=None):
    from .sae_interp_stability import analyze, save

    spec = configuration()
    root = campaign_root()
    groups = defaultdict(list)
    for path in sorted((root / "jobs").glob("*.json")):
        job = read_json(path)
        groups[group_key(job)].append(job)
    smallest = min(len(spec["grid"]["seeds"]), len(spec["variants"][1]["seeds"]))
    for key, jobs in sorted(groups.items()):
        destination = root / "stability" / key
        completed = [job for job in jobs if (root / "runs" / job["run_id"] / "result.json").exists()]
        # reference-seed screens have no seeds to compare until their other seeds are trained
        if len(jobs) < smallest or len(completed) < len(jobs):
            continue
        if (destination / "summary.json").exists():
            assert set(read_json(destination / "summary.json")["dictionaries"]) == {job["run_id"] for job in completed}, f"stability summary of {key} covers other runs"
            continue
        config = spec if len(jobs) == len(spec["grid"]["seeds"]) else {**spec, "stability": {**spec["stability"], "stable_minimum_reappearance": len(jobs) - 1, "unstable_maximum_reappearance": 0}}
        started = time.monotonic()
        summary, arrays = analyze(root, completed, config)
        save(destination, key, jobs, completed, {**summary, "stable_rule": f"reappears in at least {config['stability']['stable_minimum_reappearance']} of {len(jobs) - 1} other seeds"}, arrays, root, started)
        print("STABILITY", key, [item["stable_features"] for item in summary["seeds"]], round(summary["reference_stable_explained_variance"], 3), round(summary["pca_reference"]["sample_fvu"], 3), flush=True)
        torch.cuda.empty_cache()


@torch.inference_mode()
def region_coverage(dictionary, tokens, stable, thresholds, patient_count, bootstrap, seed):
    # same computation as sae_interp_coverage.coverage
    result, sums, counts, per_patient = {}, {}, {}, {}
    keep = torch.from_numpy(stable).cuda().float()
    for name, (values, patients) in tokens.items():
        normalized = (values - dictionary.mean) / dictionary.scale
        codes = dictionary.encode(values)
        deviation = (normalized - normalized.mean(0)).double().square().sum(1)
        full = (codes @ dictionary.weights["W_dec"] + dictionary.weights["b_dec"] - normalized).double().square().sum(1)
        kept = ((codes * keep) @ dictionary.weights["W_dec"] + dictionary.weights["b_dec"] - normalized).double().square().sum(1)
        result[name] = {"tokens": len(values), "explained_variance": float(1 - full.sum() / deviation.sum()), "stable_explained_variance": float(1 - kept.sum() / deviation.sum()),
                        "stable_activation_share": float((codes * keep).sum() / codes.sum())}
        sums[name], counts[name] = codes.double().sum(0), len(values)
        per_patient[name] = [torch.zeros(patient_count, dtype=torch.float64, device="cuda").index_add_(0, patients, value) for value in [kept, deviation]]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.zeros((bootstrap, patient_count), dtype=torch.float64, device="cuda")
    weights.scatter_add_(1, torch.randint(patient_count, (bootstrap, patient_count), generator=generator, device="cuda"), torch.ones((bootstrap, patient_count), dtype=torch.float64, device="cuda"))
    sampled = {name: 1 - (weights @ kept) / (weights @ deviation) for name, (kept, deviation) in per_patient.items()}
    difference = (sampled["nodule_0_4mm"] - sampled["far_over_10mm"]).cpu().numpy()
    overall = sum(sums.values()) / sum(counts.values())
    enrichment = (sums["nodule_0_4mm"] / counts["nodule_0_4mm"] / torch.where(overall > 0, overall, torch.ones_like(overall))).cpu().numpy()
    enrichment[overall.cpu().numpy() == 0] = 0
    enriched = {str(value): {"features": int((enrichment >= value).sum()), "stable": int(((enrichment >= value) & stable).sum())} for value in thresholds}
    return result, enriched, {"nodule_minus_far_low": float(np.percentile(difference, 2.5)), "nodule_minus_far_high": float(np.percentile(difference, 97.5)), "bootstrap": bootstrap}


def coverage(groups):
    from .sae import SpatialDictionary

    spec = configuration()
    settings = spec["nodule_coverage"]
    root = campaign_root()
    regions = [tuple(item) for item in settings["regions"]]
    output = root / "coverage"
    output.mkdir(exist_ok=True)
    cache = {}
    for group in groups:
        if (output / f"{group}.json").exists():
            continue
        summary = read_json(root / "stability" / group / "summary.json")
        with np.load(root / "stability" / group / "features.npz") as saved:
            seeds = len(saved["seeds"])
            minimum = spec["stability"]["stable_minimum_reappearance"] if seeds == len(spec["grid"]["seeds"]) else seeds - 1
            stable = saved["development_live"][0] & (saved["reappearance"][0] >= minimum)
        first = next(iter(summary["dictionaries"]))
        dictionary = SpatialDictionary(root / "runs" / first / "dictionary.pt")
        model, site, encoder = summary["model"], summary["site"], summary["encoder"]
        if (model, site) not in cache:
            chosen, seen = {}, set()
            for record in site_records(model, site):
                if record["split"] == "development" and record["PatientID"] not in seen:
                    seen.add(record["PatientID"])
                    chosen[record["PatientID"]] = record
            cache[(model, site)] = list(chosen.values())
        records = cache[(model, site)]
        tokens = {}
        for name, low, high in regions:
            parts, patients = [], []
            for index, record in enumerate(records):
                values = np.concatenate([record[encoder], record[f"near_{encoder}"]]) if "near_distance" in record else record[encoder]
                distance = np.concatenate([record["distance"], record["near_distance"]]) if "near_distance" in record else record["distance"]
                selected = (distance >= low) & (distance < high)
                parts.append(values[selected])
                patients.append(np.full(int(selected.sum()), index))
            tokens[name] = (torch.from_numpy(np.concatenate(parts).astype(np.float32)).cuda(), torch.from_numpy(np.concatenate(patients)).cuda())
        result, enriched, interval = region_coverage(dictionary, tokens, stable, settings["enrichment_thresholds"], len(records), settings["bootstrap"], settings["bootstrap_seed"])
        value = {"group": group, "run_id": first, "stable_features": int(stable.sum()), "regions": result, "nodule_enriched": enriched, "interval": interval, "patients": len(records),
                 "nodule_minus_far": result["nodule_0_4mm"]["stable_explained_variance"] - result["far_over_10mm"]["stable_explained_variance"], "created_at": timestamp(), "test_used": False}
        write_json(output / f"{group}.json", value)
        print("COVERAGE", group, value["stable_features"], {name: round(item["stable_explained_variance"], 3) for name, item in result.items()}, enriched, flush=True)
        torch.cuda.empty_cache()


# Remote: summary tables, production verdicts and figure.

def production_groups(choice, model, site):
    grid = configuration()["grid"]
    stem = f"{model}_{site}_e{grid['expansion'][0]}_k{choice['k']}"
    if choice["steps"] != configuration()["training"]["steps"]:
        suffix = f"_{'nodule75' if choice['pool'] else 'uniform'}_{choice['steps'] // 1000}k"
    else:
        suffix = "_nodule75" if choice["pool"] else ""
    output = f"_{choice['objective']['name']}" if "objective" in choice else ""
    trained = f"trained_{stem}{output}{suffix}"
    return trained, f"random_{stem}{suffix}"


def report():
    import matplotlib

    matplotlib.use("Agg")
    spec = configuration()
    root = campaign_root()
    output = root / "report"
    output.mkdir(exist_ok=True)
    choices = read_json(root / "choices.json")
    rows, groups = [], []
    for path in sorted((root / "jobs").glob("*.json")):
        job = read_json(path)
        result = read_json(root / "runs" / job["run_id"] / "result.json")
        evaluation = root / "evaluation" / f"{group_key(job)}.json"
        checks = {}
        if evaluation.exists():
            value = read_json(evaluation)
            checks = next((row for row in value["classifier"]["runs"] if row["run_id"] == job["run_id"]), {})
            text = next((row for row in value.get("text", {}).get("runs", []) if row["run_id"] == job["run_id"]), {})
            checks.update({f"text_{key}": item for key, item in text.items() if key != "run_id"})
        rows.append({"group": group_key(job), "encoder": job["encoder"], "model": job["model"], "site": job["site"], "k": job["k"], "variant": job.get("variant", "uniform"), "seed": job["seed"],
                     "steps": result["steps"], **{key: result["development"][key] for key in ["fvu", "cosine", "l0_mean", "inactive_fraction"]}, **{key: item for key, item in checks.items() if key != "run_id"}})
    table = pd.DataFrame(rows)
    table.to_csv(output / "runs.csv", index=False)
    for directory in sorted((root / "stability").iterdir()):
        summary = read_json(directory / "summary.json")
        coverage_path = root / "coverage" / f"{directory.name}.json"
        cover = read_json(coverage_path) if coverage_path.exists() else None
        groups.append({"group": directory.name, "encoder": summary["encoder"], "model": summary["model"], "site": summary["site"], "k": summary["k"], "seeds": len(summary["seeds"]), "stable_rule": summary["stable_rule"],
                       "reference_stable_features": summary["seeds"][0]["stable_features"], "mean_stable_features": float(np.mean([item["stable_features"] for item in summary["seeds"]])),
                       "reference_live_features": summary["seeds"][0]["live_features"], "reference_sample_fvu": summary["seeds"][0]["sample_fvu"], "pca_same_rank_fvu": summary["pca_reference"]["sample_fvu"],
                       "reference_stable_explained_variance": summary["reference_stable_explained_variance"],
                       **({"nodule_stable_explained_variance": cover["regions"]["nodule_0_4mm"]["stable_explained_variance"], "far_stable_explained_variance": cover["regions"]["far_over_10mm"]["stable_explained_variance"],
                           "stable_nodule_enriched_3": cover["nodule_enriched"]["3.0"]["stable"], "nodule_enriched_3": cover["nodule_enriched"]["3.0"]["features"]} if cover else {})})
    groups = pd.DataFrame(groups)
    groups.to_csv(output / "groups.csv", index=False)
    verdicts = {}
    for model, site, _ in site_list():
        choice = choices[f"{model}:{site}"]
        if choice["k"] is None:
            verdicts[f"{model}:{site}"] = {"choice": choice, "accepted": False, "shortfall": "no configuration passes the fixed-classifier gate with all 8 seeds"}
            continue
        trained, random = production_groups(choice, model, site)
        trained_row, random_row = groups.set_index("group").loc[trained], groups.set_index("group").loc[random]
        runs = table[table.group == trained]
        verdict = {"choice": choice, "production": {"trained": trained, "random": random},
                   "classifier_gate": gate_status(trained), "gate": "acceptance_revision" if revised_gate(trained) else "acceptance",
                   "classifier_mae_range": [float(runs.probability_mae.min()), float(runs.probability_mae.max())],
                   "auroc_decrease_range": [float(runs.auroc_decrease.min()), float(runs.auroc_decrease.max())],
                   "ap_decrease_range": [float(runs.ap_decrease.min()), float(runs.ap_decrease.max())], "ap_decrease_mean": float(runs.ap_decrease.mean()),
                   "reference_fvu_below_pca": bool(trained_row.reference_sample_fvu < trained_row.pca_same_rank_fvu), "reference_sample_fvu": float(trained_row.reference_sample_fvu), "pca_same_rank_fvu": float(trained_row.pca_same_rank_fvu),
                   "stable_features": {"trained": int(trained_row.reference_stable_features), "random": int(random_row.reference_stable_features)},
                   "stable_exceeds_random": bool(trained_row.reference_stable_features > random_row.reference_stable_features)}
        if "text_passes" in runs:
            verdict.update({"all_seeds_pass_text": bool(runs.text_passes.all()), "text_image_cosine_range": [float(runs.text_image_cosine.min()), float(runs.text_image_cosine.max())],
                            "text_median_spearman_range": [float(runs.text_median_score_spearman.min()), float(runs.text_median_score_spearman.max())]})
        verdict["accepted"] = bool(verdict["classifier_gate"] == "passed" and verdict.get("all_seeds_pass_text", True) and verdict["reference_fvu_below_pca"] and verdict["stable_exceeds_random"])
        verdicts[f"{model}:{site}"] = verdict
    # the interpretation materials: per site the production dictionaries (8 seeds), their stable-feature analysis and the random-weight control
    production = {}
    for model, site, _ in site_list():
        verdict = verdicts[f"{model}:{site}"]
        if "production" not in verdict:
            continue
        trained, random = verdict["production"]["trained"], verdict["production"]["random"]
        members = sorted((job for job in (read_json(path) for path in (root / "jobs").glob("*.json")) if group_key(job) == trained), key=lambda job: job["seed"])
        summary = read_json(root / "stability" / trained / "summary.json")
        production[f"{model}:{site}"] = {
            "accepted": verdict["accepted"], "group": trained, "random_control": random, "k": verdict["choice"]["k"], "expansion": spec["grid"]["expansion"][0],
            "objective": verdict["choice"].get("objective", {}).get("name", "token loss"), "steps": verdict["choice"]["steps"], "reference_seed": spec["grid"]["reference_seed"],
            "site": spec["models"][model][site],
            "dictionaries": [{"seed": job["seed"], "run_id": job["run_id"], "path": f"runs/{job['run_id']}/dictionary.pt", "sha256": read_json(root / "runs" / job["run_id"] / "result.json")["dictionary_sha256"]} for job in members],
            "stability": {"directory": f"stability/{trained}", "stable_rule": summary.get("stable_rule"), "stable_features": [item["stable_features"] for item in summary["seeds"]],
                          "reference_stable_explained_variance": summary["reference_stable_explained_variance"]},
            "coverage": f"coverage/{trained}.json", "evaluation": f"evaluation/{trained}.json"}
    write_json(output / "production.json", {"campaign": spec["campaign"], "root": "paths are relative to the campaign directory", "sites": production, "created_at": timestamp(), "test_used": False})
    write_json(output / "summary.json", {"campaign": spec["campaign"], "verdicts": verdicts, "quality_rule": spec["quality"], "selection": spec["selection"], "code_sha256": sha256(Path(__file__)),
                                         "config_sha256": sha256(CONFIG_PATH), "created_at": timestamp(), "test_used": False})
    figure(table, groups, choices, output)
    gate_figure(table, choices, output)
    for key, verdict in verdicts.items():
        if "shortfall" in verdict:
            print("VERDICT", key, verdict["accepted"], verdict["shortfall"], flush=True)
            continue
        print("VERDICT", key, verdict["accepted"], verdict["choice"]["k"], verdict["choice"].get("objective", {}).get("name"), verdict["choice"]["steps"], verdict["choice"]["pool"], verdict["stable_features"],
              round(verdict["reference_sample_fvu"], 3), round(verdict["pca_same_rank_fvu"], 3), verdict["classifier_mae_range"], flush=True)


def quality_entries(choices, model, site):
    # (label, group, kind) compared per site: the token-loss dictionary at the chosen k, the production dictionary, the from-scratch head-read comparison and the random-weight control
    choice = choices[f"{model}:{site}"]
    trained, random = production_groups(choice, model, site)
    stem = f"trained_{model}_{site}_e{configuration()['grid']['expansion'][0]}"
    token = f"{stem}_k{choice['k']}"
    if trained == token:
        entries = [(f"production:\ntoken loss\nk{choice['k']}, 48k", token, "token")]
    else:
        entries = [(f"token loss\nk{choice['k']}, 48k", token, "token"), (f"production:\nfine-tuned\n{trained.removeprefix(stem + '_').replace('_', ' ')}", trained, "ft" if "objective" in choice else "token")]
    scratch = {"flexict_3d:f32_b10": "k32_read_w3", "flexict_vlm:f40_b08": "k16_read_w1"}.get(f"{model}:{site}")
    if scratch:
        entries.append((f"head-read\nfrom scratch\n{scratch.replace('_', ' ')}", f"{stem}_{scratch}", "read"))
    entries.append((f"random-weight\ncontrol\n{random.removeprefix('random_' + model + '_' + site + '_e4_').replace('_', ' ')}", random, "random"))
    return entries


def figure(table, groups, choices, output):
    # per site: stable features of every seed and the reference seed's explained variance for the dictionaries that decide the production verdict
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, surface, grid_color = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"
    colors = {"token": "#2a78d6", "read": "#1baf7a", "ft": "#eda100", "random": "#898781"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": muted, "axes.labelcolor": ink})
    names = {"flexict_3d:f32_b10": "FlexiCT-3D, 32 mm field, block 10", "flexict_vlm:f40_b08": "FlexiCT-3D-VLM, 40 mm field, block 8", "flexict_vlm:w20_b16": "FlexiCT-3D-VLM, whole CT 2.0 mm, block 16"}
    sites = site_list()
    available = set(groups.group)
    entries = {f"{model}:{site}": [entry for entry in quality_entries(choices, model, site) if entry[1] in available] for model, site, _ in sites}
    fig, axes = plt.subplots(2, len(sites), figsize=(15, 7.5), facecolor=surface, squeeze=False, gridspec_kw={"width_ratios": [max(1, len(entries[f"{model}:{site}"])) for model, site, _ in sites]})
    for column, (model, site, _) in enumerate(sites):
        top, bottom = axes[0, column], axes[1, column]
        rows = entries[f"{model}:{site}"]
        for position, (label, group, kind) in enumerate(rows):
            summary = read_json(campaign_root() / "stability" / group / "summary.json")
            counts = [item["stable_features"] for item in summary["seeds"]]
            offsets = np.linspace(-0.2, 0.2, len(counts))
            top.scatter(position + offsets, counts, s=20, color=colors[kind], edgecolors=surface, linewidths=0.6, zorder=3)
            row = groups.set_index("group").loc[group]
            bottom.bar(position - 0.17, 1 - row.reference_sample_fvu, width=0.32, color=colors[kind], zorder=2)
            bottom.bar(position + 0.17, row.reference_stable_explained_variance, width=0.32, color=colors[kind], alpha=0.45, hatch="///", edgecolor=surface, zorder=2)
            bottom.hlines(1 - row.pca_same_rank_fvu, position - 0.36, position + 0.36, color=ink, linewidth=1.3, zorder=4)
        for axis in [top, bottom]:
            axis.set_xticks(range(len(rows)), [item[0] for item in rows], fontsize=7.5)
            axis.set_xlim(-0.6, len(rows) - 0.4)
            axis.grid(axis="y", color=grid_color, linewidth=0.7)
            axis.set_axisbelow(True)
            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
        top.set_ylabel("stable features per seed")
        top.set_ylim(bottom=0)
        bottom.set_ylabel("explained variance, reference seed")
        bottom.axhline(0, color=muted, linewidth=0.8)
        top.set_title(names[f"{model}:{site}"], loc="left", fontsize=9)
    handles = [Line2D([], [], color=muted, marker="s", linestyle="none", markersize=7, label="bottom, solid: all features (1 − FVU)"),
               Line2D([], [], color=muted, marker="s", linestyle="none", markersize=7, alpha=0.45, label="bottom, hatched: stable features only"),
               Line2D([], [], color=ink, linewidth=1.3, label="bottom, black bar: same-rank PCA (1 − FVU)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.6)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"sae_quality.{suffix}", dpi=170, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def gate_figure(table, choices, output):
    # per site and configuration: the fixed-classifier gate (probability MAE, AP decrease) and the error inside the subspace the head reads
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, surface, grid_color, band = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9", "#f0efec"
    styles = {"token": ("#2a78d6", "o", "token loss"), "kl": ("#eb6834", "^", "token loss + bag KL of the head output"), "read": ("#1baf7a", "s", "head-read objective from scratch"),
              "ft": ("#eda100", "D", "head-read fine-tuning of the token-loss dictionary")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": muted, "axes.labelcolor": ink})
    acceptance = configuration()["acceptance"]
    names = {"flexict_3d:f32_b10": "FlexiCT-3D, 32 mm field, block 10", "flexict_vlm:f40_b08": "FlexiCT-3D-VLM, 40 mm field, block 8", "flexict_vlm:w20_b16": "FlexiCT-3D-VLM, whole CT 2.0 mm, block 16"}
    trained = table[(table.encoder == "trained") & table.probability_mae.notna()]
    sites = site_list()
    # configurations: uniform-sampling dictionaries of each objective, at 48,000 steps and, where trained, as 192,000-step productions
    patterns = [("token", r"uniform|uniform_192k"), ("kl", r"e2e_w\d+"), ("read", r"read_w\d+"), ("ft", r"ft_w\d+|ft_w\d+_uniform_192k")]
    width = [max(1, trained[(trained.model == model) & (trained.site == site) & trained.variant.str.fullmatch("|".join(item[1] for item in patterns))].group.nunique()) for model, site, _ in sites]
    fig, axes = plt.subplots(3, len(sites), figsize=(20, 9.5), facecolor=surface, squeeze=False, gridspec_kw={"width_ratios": width})
    weights = lambda variant: int(variant.split("_w")[1].split("_")[0]) if "_w" in variant else 0
    for column, (model, site, _) in enumerate(sites):
        rows = trained[(trained.model == model) & (trained.site == site)]
        configurations = []
        for kind, pattern in patterns:
            chosen_rows = rows[rows.variant.str.fullmatch(pattern)]
            for (k, variant), _ in sorted(chosen_rows.groupby(["k", "variant"]), key=lambda item: (item[0][0], weights(item[0][1]), item[0][1].endswith("192k"))):
                long = "\n192k" if variant.endswith("192k") else ""
                configurations.append((kind, k, variant, f"k{k}{long}" if kind == "token" else f"k{k}\nw{weights(variant)}{long}"))
        choice = choices[f"{model}:{site}"]
        production = production_groups(choice, model, site)[0]
        stem = f"trained_{model}_{site}_e{configuration()['grid']['expansion'][0]}_k{choice['k']}"
        chosen_variant = production.removeprefix(stem + "_") if production != stem else "uniform"
        revised = f"{model}:{site}" in configuration()["acceptance_revision"]["sites"]
        for row_index, (measure, label, limit) in enumerate([("probability_mae", "probability MAE", acceptance["max_probability_mae"]), ("ap_decrease", "AP decrease", acceptance["max_ap_decrease"]), ("head_space_fvu", "FVU (log scale)", None)]):
            axis = axes[row_index, column]
            for position, (kind, k, variant, _) in enumerate(configurations):
                color, marker, _ = styles[kind]
                values = rows[(rows.k == k) & (rows.variant == variant)]
                if choice["k"] == k and variant == chosen_variant:
                    axis.axvspan(position - 0.45, position + 0.45, color=band, zorder=0)
                offsets = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
                axis.scatter(position + offsets, values[measure], s=22, color=color, marker=marker, edgecolors=surface, linewidths=0.8, zorder=3)
                if measure == "ap_decrease" and revised and len(values) > 1:
                    axis.hlines(values[measure].mean(), position - 0.3, position + 0.3, color=ink, linewidth=1.4, zorder=4)
                if measure == "head_space_fvu":
                    axis.scatter(position + offsets, values["fvu"], s=22, facecolors="none", edgecolors=color, marker=marker, linewidths=1.0, zorder=3)
            if limit is not None:
                axis.axhline(limit, color=muted, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
                gate_label = f"gate {limit} on the seed mean (black bar)" if measure == "ap_decrease" and revised else f"gate {limit}"
                axis.annotate(gate_label, xy=(0, limit), xycoords=("axes fraction", "data"), xytext=(2, 2), textcoords="offset points", color=muted, fontsize=7.5, ha="left", va="bottom")
            if measure == "head_space_fvu":
                axis.set_yscale("log")
            kinds = [item[0] for item in configurations]
            for position in [index for index in range(1, len(kinds)) if kinds[index] != kinds[index - 1]]:
                axis.axvline(position - 0.5, color=grid_color, linewidth=1.0, zorder=1)
            if row_index == 2:
                for kind in dict.fromkeys(kinds):
                    positions = [index for index, item in enumerate(kinds) if item == kind]
                    axis.annotate({"token": "token loss", "kl": "bag KL", "read": "head-read, weight w", "ft": "fine-tuned, weight w"}[kind], xy=(np.mean(positions), 0), xycoords=("data", "axes fraction"),
                                  xytext=(0, -42), textcoords="offset points", ha="center", va="top", color=muted, fontsize=8)
            axis.set_xticks(range(len(configurations)), [item[3] for item in configurations], fontsize=7)
            axis.set_xlim(-0.6, len(configurations) - 0.4)
            axis.set_ylabel(label)
            axis.grid(axis="y", color=grid_color, linewidth=0.7)
            axis.set_axisbelow(True)
            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
        axes[0, column].set_title(f"{names[f'{model}:{site}']}\nshaded: production dictionaries", loc="left", fontsize=9)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=6, label=name) for color, marker, name in styles.values()]
    handles += [Line2D([], [], color=muted, marker="o", linestyle="none", markersize=6, label="bottom row, filled: head-space FVU"),
                Line2D([], [], color=muted, marker="o", markerfacecolor="none", linestyle="none", markersize=6, label="bottom row, hollow: overall FVU")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.2)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"classifier_gate.{suffix}", dpi=170, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


# Remote: selection by the rules written in the configuration.

def revised_gate(group):
    # sites under acceptance_revision judge AP on the mean over the seeds (user decision 2026-10-22); the others keep the per-seed gate
    site = next(f"{model}:{site}" for model, site, _ in site_list() if group.startswith(f"trained_{model}_{site}_"))
    return site in configuration()["acceptance_revision"]["sites"]


def gate_status(group):
    # "passed": every seed evaluated and the gate met; "failed": the gate can no longer be met; "open": seeds missing and nothing failed yet
    spec = configuration()
    limits, seeds = spec["acceptance"], len(spec["grid"]["seeds"])
    result = read_json(campaign_root() / "evaluation" / f"{group}.json")
    rows = result["classifier"]["runs"]
    if not all(row["probability_mae"] <= limits["max_probability_mae"] and row["auroc_decrease"] <= limits["max_auroc_decrease"] for row in rows):
        return "failed"
    if not all(row["passes"] for row in result.get("text", {}).get("runs", [])):
        return "failed"
    if revised_gate(group):
        if len(rows) < seeds:
            return "open"
        return "passed" if np.mean([row["ap_decrease"] for row in rows]) <= limits["max_ap_decrease"] else "failed"
    if any(row["ap_decrease"] > limits["max_ap_decrease"] for row in rows):
        return "failed"
    return "passed" if len(rows) >= seeds else "open"


def passes(group):
    return gate_status(group) == "passed"


def reference_passes(group):
    result = read_json(campaign_root() / "evaluation" / f"{group}.json")
    limits = configuration()["acceptance"]
    row = result["classifier"]["runs"][0]
    text = result.get("text", {}).get("runs", [])
    seed_passes = row["probability_mae"] <= limits["max_probability_mae"] and row["auroc_decrease"] <= limits["max_auroc_decrease"]
    return bool(seed_passes and (revised_gate(group) or row["ap_decrease"] <= limits["max_ap_decrease"]) and (text[0]["passes"] if text else True))


def four_seed_stable(group_jobs):
    from .sae_interp_stability import analyze

    spec = configuration()
    seeds = configuration()["variants"][1]["seeds"]
    jobs = [job for job in group_jobs if job["seed"] in seeds]
    summary, _ = analyze(campaign_root(), jobs, {**spec, "stability": {**spec["stability"], "stable_minimum_reappearance": len(seeds) - 1, "unstable_maximum_reappearance": 0}})
    torch.cuda.empty_cache()
    return summary["seeds"][0]["stable_features"]


def select(stage, sites=None):
    # sites: optional subset ("model:site") decided now; the other sites keep their recorded choices
    spec = configuration()
    root = campaign_root()
    grid = spec["grid"]
    jobs = defaultdict(list)
    for path in sorted((root / "jobs").glob("*.json")):
        job = read_json(path)
        jobs[group_key(job)].append(job)
    choices = read_json(root / "choices.json") if (root / "choices.json").exists() else {}
    for model, site, _ in site_list():
        key = f"{model}:{site}"
        if sites is not None and key not in sites:
            continue
        base = lambda encoder, k, variant=None: f"{encoder}_{model}_{site}_e{grid['expansion'][0]}_k{k}" + (f"_{variant}" if variant else "")
        if stage == "k":
            evaluated = [k for k in grid["k"] + spec["grid_extension"]["k"] if (root / "evaluation" / f"{base('trained', k)}.json").exists()]
            passing = [k for k in evaluated if passes(base("trained", k))]
            choices[key] = {"k": min(passing), "passing_k": passing, "evaluated_k": evaluated, "rule": spec["selection"]["k"]} if passing else {"k": None, "evaluated_k": evaluated, "downstream": True}
            continue
        if stage == "downstream":
            if choices[key]["k"] is None:
                options = [(k, weight) for k in spec["downstream"]["k"] for weight in spec["downstream"]["weights"] if (root / "evaluation" / f"{base('trained', k, f'e2e_w{weight}')}.json").exists()]
                passing = [(k, weight) for k, weight in options if passes(base("trained", k, f"e2e_w{weight}"))]
                choices[key].update({"passing_downstream": passing, "evaluated_downstream": options})
                if passing:
                    k, weight = min(passing)
                    choices[key].update({"k": k, "objective": {"name": f"e2e_w{weight}", "training": {"output_loss": {"weight": weight}}}, "rule": spec["downstream"]["rule"]})
            continue
        if stage == "head_finetune":
            # the from-scratch head-read dictionaries are a recorded comparison, reported from their evaluation and stability files
            if choices[key]["k"] is None:
                prefix = "ft"
                settings = spec["head_read_finetune"]
                group = lambda k, weight: base("trained", k, f"{prefix}_w{weight}")
                configurations = [(k, weight) for k in settings["k"] for weight in settings["weights"]]
                options = [item for item in configurations if (root / "evaluation" / f"{group(*item)}.json").exists()]
                assert options == configurations, f"{key}: {stage} screens not all evaluated {sorted(set(configurations) - set(options))}"
                screened = [item for item in options if reference_passes(group(*item))]
                seeded = [item for item in screened if len(read_json(root / "evaluation" / f"{group(*item)}.json")["classifier"]["runs"]) == len(grid["seeds"])]
                # a configuration whose gate can no longer be met is ruled out; the first remaining one must have passed with all seeds evaluated
                status = {item: gate_status(group(*item)) for item in screened}
                failed = [item for item in screened if status[item] == "failed"]
                passing = [item for item in screened if status[item] == "passed"]
                first = next((item for item in screened if status[item] != "failed"), None)
                assert first is None or first in passing, f"{key}: {first} is still open; its other seeds are not all evaluated"
                choices[key]["head_read_finetune"] = {"evaluated": options, "screen_passing": screened, "failed": failed, "seeded": seeded, "passing": passing, "first_passing": first,
                                                      "rule": settings["rule"], "gate": "acceptance_revision" if revised_gate(group(*configurations[0])) else "acceptance"}
                if first is not None:
                    k, weight = first
                    training = {**{name: jobs[group(k, weight)][0]["training"][name] for name in ["pooled_loss", "head_space"]}, "steps": settings["steps"]}
                    choices[key].update({"k": k, "objective": {"name": f"ft_w{weight}", "training": training, "init": settings["source"]}, "rule": settings["rule"]})
            continue
        if choices[key]["k"] is None:
            continue
        if stage == "production":
            # selection.length_revision: a longer production replaces the 48,000-step dictionaries only when all its seeds pass the gate
            longer = choices[key].get("rejected_steps", choices[key].get("steps", spec["training"]["steps"]))
            if longer != spec["training"]["steps"]:
                trained, _ = production_groups({**choices[key], "steps": longer}, model, site)
                status = gate_status(trained)
                choices[key]["production_check"] = {"group": trained, "classifier_gate": status, "rule": spec["selection"]["length_revision"]}
                choices[key].pop("rejected_steps", None)
                choices[key]["steps"] = longer
                if status != "passed":
                    choices[key].update({"rejected_steps": longer, "steps": spec["training"]["steps"]})
            continue
        k = choices[key]["k"]
        prefix = choices[key]["objective"]["name"] if "objective" in choices[key] else None
        uniform, pooled, long = base("trained", k, prefix), base("trained", k, f"{prefix}_nodule75" if prefix else "nodule75"), base("trained", k, f"{prefix}_tokens16x" if prefix else "tokens16x")
        stable_48k, stable_192k = four_seed_stable(jobs[uniform]), four_seed_stable(jobs[long])
        steps = 192000 if stable_192k >= 1.2 * stable_48k and reference_passes(long) else spec["training"]["steps"]
        coverage_value = {name: read_json(root / "coverage" / f"{name}.json") for name in [uniform, pooled, base("random", k), base("random", k, "nodule75")]}
        nodule = {name: value["nodule_enriched"]["3.0"]["stable"] for name, value in coverage_value.items()}
        clearly = lambda value, other: value >= 1.5 * other and value >= other + 5
        pool = (passes(pooled) and clearly(nodule[pooled], nodule[uniform]) and clearly(nodule[pooled], nodule[base("random", k, "nodule75")])
                and coverage_value[pooled]["regions"]["nodule_0_4mm"]["stable_explained_variance"] > coverage_value[uniform]["regions"]["nodule_0_4mm"]["stable_explained_variance"])
        choices[key].update({"steps": steps, "pool": bool(pool), "four_seed_stable": {"48000": stable_48k, "192000": stable_192k}, "stable_nodule_features": nodule,
                             "nodule_region_stable_explained_variance": {name: value["regions"]["nodule_0_4mm"]["stable_explained_variance"] for name, value in coverage_value.items()},
                             "pool_passes_classifier": passes(pooled), "rules": {"length": spec["selection"]["length"], "pool": spec["selection"]["pool"], "clearly_exceed": spec["selection"]["clearly_exceed"]}})
    write_json(root / "choices.json", choices)
    print("CHOICES", {key: {name: value for name, value in item.items() if name in ["k", "steps", "pool", "four_seed_stable", "stable_nodule_features"]} for key, item in choices.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["distances", "text", "extract", "matrices", "jobs", "worker", "yield", "stability", "evaluate", "coverage", "select", "report"], required=True)
    parser.add_argument("--select", choices=["k", "downstream", "head_finetune", "variants", "production"])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--phase", choices=["grid", "extension", "downstream", "head_screen", "head_seeds", "finetune_screen", "finetune_seeds", "variants", "production"])
    parser.add_argument("--choices", type=Path, help="selection JSON with the chosen k per site")
    parser.add_argument("--gpu")
    parser.add_argument("--name")
    parser.add_argument("--groups", help="comma-separated group names")
    parser.add_argument("--sites", help="comma-separated model:site keys for select")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage == "distances":
        distances(args.workers)
    elif args.stage == "text":
        text_embeddings()
    elif args.stage == "extract":
        extract(args.part, args.parts)
    elif args.stage == "matrices":
        matrices()
    elif args.stage == "jobs":
        write_jobs(args.phase, read_json(args.choices) if args.choices else None)
    elif args.stage == "worker":
        worker(args.gpu, args.name)
    elif args.stage == "yield":
        yield_control()
    elif args.stage == "stability":
        stability()
    elif args.stage == "evaluate":
        evaluate(args.groups.split(","))
    elif args.stage == "coverage":
        coverage(args.groups.split(","))
    elif args.stage == "select":
        select(args.select, args.sites.split(",") if args.sites else None)
    else:
        report()


if __name__ == "__main__":
    main()

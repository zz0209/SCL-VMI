import argparse
import gc
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from nibabel.processing import resample_from_to
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json


CONFIG = ROOT / "configs/flexict_luna25.json"
PIPELINE = ROOT / "configs/encoder_pipeline.json"
SOURCES = ["src/sclvmi/flexict_luna25.py", "src/sclvmi/models.py", "src/sclvmi/frozen_heads.py", "configs/flexict_luna25.json", "configs/model_sources.json"]
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
CROP = np.array([128, 128, 64])
PATCH = 8
METRICS = ["auroc", "average_precision", "brier", "log_loss"]


def specification():
    return read_json(CONFIG)


def suite_root():
    return Path(context()[0]["runs"]) / specification()["suite"]


# Local preparation: model inputs rebuilt from the official whole CTs.

def official_table(luna_root):
    table = pd.read_csv(luna_root / "zenodo" / "14673658" / "LUNA25_Public_Training_Development_Data.csv")
    table["identifier"] = table.PatientID.astype(str) + "_" + table.LesionID.astype(str) + "_" + table.StudyDate.astype(str)
    assert table.identifier.is_unique
    return table.set_index("identifier")


def cases():
    from .data import load_manifest

    storage, _, dataset = context()
    luna_root = Path(storage["luna25_root"])
    manifest = load_manifest()
    manifest = manifest.sort_values("identifier").reset_index(drop=True)
    official = official_table(luna_root).loc[manifest.identifier]
    assert (official.label.to_numpy() == manifest.label.to_numpy()).all() and (official.SeriesInstanceUID.to_numpy() == manifest.SeriesInstanceUID.to_numpy()).all()
    frame = manifest[["identifier", "PatientID", "SeriesInstanceUID", "label", "fold", "split"]].copy()
    frame[["CoordX", "CoordY", "CoordZ"]] = official[["CoordX", "CoordY", "CoordZ"]].to_numpy(dtype=float)
    frame["scan"] = [str(luna_root / "images" / "luna25_images" / f"{series}.mha") for series in frame.SeriesInstanceUID]
    frame["provider_crop"] = [str(dataset / "imagesTr" / f"{identifier}_0000.nii.gz") for identifier in frame.identifier]
    return frame


def index_affine(scan):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(scan.GetDirection()).reshape(3, 3) @ np.diag(scan.GetSpacing())
    matrix[:3, 3] = scan.GetOrigin()
    return matrix


def provider_crop(volume, matrix, start):
    # 128 x 128 x 64 native voxels around the rounded coordinate, -1024 outside the scan, written and read back as NIfTI like the provider files
    crop = np.full(tuple(CROP), -1024, dtype=volume.dtype)
    low, high = np.maximum(start, 0), np.minimum(start + CROP, volume.shape)
    crop[tuple(slice(a - s, b - s) for a, b, s in zip(low, high, start))] = volume[tuple(slice(a, b) for a, b in zip(low, high))]
    shift = np.eye(4)
    shift[:3, 3] = start
    return nib.Nifti1Image.from_bytes(nib.Nifti1Image(crop, RAS_TO_LPS @ matrix @ shift).to_bytes())


def physical_field(image, centre, size, spacing):
    # frozen_features.physical_crop with an explicit centre: LPS target grid, linear interpolation, -1024 outside, (S, P, L) array axes
    target = np.eye(4)
    target[:3, :3] = np.diag(np.asarray([spacing] * 3) * np.array([-1, -1, 1]))
    target[:3, 3] = centre - target[:3, :3] @ (np.asarray([size] * 3) // 2)
    sampled = resample_from_to(image, ((size,) * 3, target), order=1, mode="constant", cval=-1024)
    return np.array(sampled.dataobj, dtype=np.float32).transpose(2, 1, 0)


def official_volume(scan, spacing):
    # downstream/vlm/dataset.py CT3D_CLIP.resample_to_isotropic and the clamp before z-scoring
    if not all(abs(value - spacing) <= 1e-4 + 1e-4 * spacing for value in scan.GetSpacing()):
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetOutputSpacing((spacing,) * 3)
        resample.SetSize([int(round(size * step / spacing)) for size, step in zip(scan.GetSize(), scan.GetSpacing())])
        resample.SetOutputOrigin(scan.GetOrigin())
        resample.SetOutputDirection(scan.GetDirection())
        resample.SetOutputPixelType(scan.GetPixelID())
        scan = resample.Execute(scan)
    clamped = torch.from_numpy(sitk.GetArrayFromImage(scan)).to(torch.float32).clamp(-1000, 1000)
    return scan, clamped.numpy().astype(np.int16), float(clamped.mean()), float(clamped.std())


def window_placement(volume_shape, nodule, view):
    # official window: padding to at least the window size, then the MONAI centre crop; the readout window moves only for nodules within margin_mm of its border
    size = np.asarray(view["window"])
    padded = np.maximum(volume_shape, size)
    pad = (padded - volume_shape) // 2
    position = nodule + pad
    official = (padded - size) // 2
    margin = view["margin_mm"] / view["spacing_mm"]
    start = np.clip(official, np.ceil(position + margin - size).astype(int), np.floor(position - margin).astype(int))
    start = np.clip(start, 0, padded - size)
    tokens = (position - start) / PATCH
    readout = np.clip(np.rint(tokens - view["readout_tokens"] / 2).astype(int), 0, size // PATCH - view["readout_tokens"])
    return official, start, pad, position - start, readout


def cut_window(volume, start, pad, size, fill):
    window = np.full(tuple(size), fill, dtype=np.int16)
    low = np.maximum(start - pad, 0)
    high = np.minimum(start - pad + size, volume.shape)
    window[tuple(slice(a - (s - p), b - (s - p)) for a, b, s, p in zip(low, high, start, pad))] = volume[tuple(slice(a, b) for a, b in zip(low, high))]
    return window


def prepare_scan(path, rows, spec):
    from .frozen_features import physical_crop

    scan = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(scan).transpose(2, 1, 0)
    matrix = index_affine(scan)
    whole = nib.Nifti1Image(volume, RAS_TO_LPS @ matrix)
    fields, centres = [], []
    for row in rows:
        # luna25-baseline-public preprocessing/extract_nodules.py: TransformPhysicalPointToIndex (ITK rounds halves up), then index - extent
        start = np.asarray(scan.TransformPhysicalPointToIndex((float(row["CoordX"]), float(row["CoordY"]), float(row["CoordZ"])))) - CROP // 2
        crop = provider_crop(volume, matrix, start)
        centre = nib.affines.apply_affine(crop.affine, np.asarray(crop.shape) // 2)
        values = {"identifier": row["identifier"]}
        for name, view in spec["views"].items():
            if view["kind"] != "field":
                continue
            spacing = view["field_mm"] / view["shape"][0]
            field = physical_field(crop if name in ["f32", "f40"] else whole, centre, view["shape"][0], spacing)
            if name in ["f32", "f40"]:
                # fields equal the baseline suites wherever the provider crop equals the official scan; corrupted provider crops must differ
                reference = physical_crop(row["provider_crop"], view["shape"][::-1], [spacing] * 3, "LPS")[0].permute(2, 1, 0).numpy()
                assert np.array_equal(reference, field) == row["provider_crop_equal"], (row["identifier"], name, float(np.abs(reference - field).max()))
            # resampling int16 scans returns int16 values, so int16 storage is lossless
            values[name] = field.astype(np.int16)
            assert np.array_equal(values[name], field), (row["identifier"], name)
        fields.append(values)
        centres.append(nib.affines.apply_affine(RAS_TO_LPS, centre))
    windows = {}
    for name, view in spec["views"].items():
        if view["kind"] != "whole":
            continue
        image, clamped, mean, std = official_volume(scan, view["spacing_mm"])
        placed, records = {}, []
        for centre in centres:
            nodule = np.asarray(image.TransformPhysicalPointToContinuousIndex(tuple(float(value) for value in centre)))[::-1]
            official, start, pad, local, readout = window_placement(np.asarray(clamped.shape), nodule, view)
            for key in [tuple(official.tolist()), tuple(start.tolist())]:
                if key not in placed:
                    placed[key] = {"window": cut_window(clamped, np.asarray(key), pad, np.asarray(view["window"]), int(clamped.min())), "start": np.asarray(key), "official": key == tuple(official.tolist())}
            records.append({"window": list(placed).index(tuple(start.tolist())), "official_window": list(placed).index(tuple(official.tolist())), "readout": readout, "nodule": local.astype(np.float32)})
        windows[name] = {"windows": list(placed.values()), "records": records, "mean": mean, "std": std, "shape": clamped.shape}
    return path, fields, windows


def write_batch(path, number, results, spec):
    temporary = path.with_suffix(".tmp")
    options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    fields = [values for _, items, _ in results for values in items]
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update({"batch": number, "created_at": timestamp(), "config_sha256": sha256(CONFIG), "code_sha256": sha256(Path(__file__))})
        handle.create_dataset("identifier", data=np.array([values["identifier"] for values in fields], dtype="S"))
        for name, view in spec["views"].items():
            if view["kind"] == "field":
                data = np.stack([values[name] for values in fields])
                handle.create_dataset(name, data=data, chunks=(1, *data.shape[1:]), **options)
                continue
            group = handle.create_group(name)
            offset, windows, scans, means, stds, starts, official, window_index, official_index, readout, nodule = 0, [], [], [], [], [], [], [], [], [], []
            for scan, _, views in results:
                item = views[name]
                windows += [entry["window"] for entry in item["windows"]]
                scans += [Path(scan).stem] * len(item["windows"])
                means += [item["mean"]] * len(item["windows"])
                stds += [item["std"]] * len(item["windows"])
                starts += [entry["start"] for entry in item["windows"]]
                official += [entry["official"] for entry in item["windows"]]
                window_index += [offset + record["window"] for record in item["records"]]
                official_index += [offset + record["official_window"] for record in item["records"]]
                readout += [record["readout"] for record in item["records"]]
                nodule += [record["nodule"] for record in item["records"]]
                offset += len(item["windows"])
            group.create_dataset("windows", data=np.stack(windows), chunks=(1, *view["window"]), **options)
            for key, value in {"scan": np.array(scans, dtype="S"), "mean": np.array(means), "std": np.array(stds), "start": np.stack(starts), "official": np.array(official),
                               "window_index": np.array(window_index), "official_index": np.array(official_index), "readout": np.stack(readout), "nodule": np.stack(nodule)}.items():
                group.create_dataset(key, data=value)
    temporary.replace(path)


def prepare(workers, smoke):
    storage, _, _ = context()
    spec = specification()
    frame = cases()
    check = pd.read_csv(Path(storage["runs"]) / spec["crop_check_run"] / "cases.csv").set_index("identifier")
    assert check.aligned.all() and set(frame.identifier) <= set(check.index)
    frame["provider_crop_equal"] = check.loc[frame.identifier, "mismatched_voxels"].to_numpy() == 0
    if smoke:
        # smoke scans per split and label, so that the smoke heads see both classes
        series = [name for split in ["train", "development", "test"] for label in [0, 1]
                  for name in sorted(frame[(frame.split == split) & (frame.label == label)].SeriesInstanceUID.unique())[:smoke]]
        frame = frame[frame.SeriesInstanceUID.isin(series)]
    output = Path(storage["preprocessed"]) / spec["suite"] / ("smoke" if smoke else "inputs")
    output.mkdir(parents=True, exist_ok=True)
    batches = []
    for prefix, part in [("batch", frame[frame.split != "test"]), ("test_batch", frame[frame.split == "test"])]:
        current = []
        for item in [(scan, rows.to_dict("records")) for scan, rows in part.groupby("scan", sort=True)]:
            current.append(item)
            if sum(len(rows) for _, rows in current) >= spec["batch_cases"]:
                batches.append((f"{prefix}_{sum(name.startswith(prefix + '_') for name, _ in batches):03d}.h5", current))
                current = []
        if current:
            batches.append((f"{prefix}_{sum(name.startswith(prefix + '_') for name, _ in batches):03d}.h5", current))
    rows = []
    for name, batch in batches:
        rows += [{**{key: row[key] for key in ["identifier", "PatientID", "SeriesInstanceUID", "label", "fold", "split", "provider_crop_equal"]}, "batch": name, "row": index}
                 for index, row in enumerate(row for _, items in batch for row in items)]
    table = pd.DataFrame(rows)
    assert table.identifier.is_unique and len(table) == len(frame)
    table.to_csv(output / "cases.csv", index=False)
    scans = sum(len(batch) for _, batch in batches)
    print(f"prepare: {len(frame)} nodules, {scans} scans, {len(batches)} batches (test batches last), {workers} workers", flush=True)
    started, done = time.monotonic(), 0
    with ProcessPoolExecutor(workers) as pool:
        for number, (name, batch) in enumerate(batches):
            path = output / name
            if not path.exists():
                results = list(pool.map(prepare_scan, [scan for scan, _ in batch], [items for _, items in batch], [spec] * len(batch)))
                write_batch(path, number, results, spec)
            done += len(batch)
            elapsed = time.monotonic() - started
            print(f"prepare: {name} ({number + 1}/{len(batches)}), {done}/{scans} scans, {path.stat().st_size / 1e9:.2f} GB, {elapsed / 60:.1f} min, eta {elapsed / done * (scans - done) / 60:.1f} min", flush=True)
    summary = {"suite": spec["suite"], "nodules": len(table), "scans": scans, "batches": [{"name": name, "bytes": (output / name).stat().st_size, "sha256": sha256(output / name)} for name, _ in batches],
               "cases_sha256": sha256(output / "cases.csv"), "splits": table.split.value_counts().to_dict(), "config_sha256": sha256(CONFIG),
               "provider_crops_differing_from_official_scan": table.loc[~table.provider_crop_equal, "identifier"].tolist(),
               "code_sha256": sha256(Path(__file__)), "created_at": timestamp()}
    windows = {name: {"windows": 0, "official_windows": 0, "nodules": 0, "nodules_read_from_moved_window": 0} for name, view in spec["views"].items() if view["kind"] == "whole"}
    for item in summary["batches"]:
        with h5py.File(output / item["name"], "r") as handle:
            for name in windows:
                windows[name]["windows"] += len(handle[name]["official"])
                windows[name]["official_windows"] += int(handle[name]["official"][:].sum())
                windows[name]["nodules"] += len(handle[name]["window_index"])
                windows[name]["nodules_read_from_moved_window"] += int((handle[name]["window_index"][:] != handle[name]["official_index"][:]).sum())
    summary["whole_windows"] = windows
    write_json(output / "inputs.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "batches"}), flush=True)


# Remote extraction: frozen FlexiCT on the prepared inputs.

def field_input(array):
    tensor = torch.from_numpy(array.astype(np.float32)).clamp(-1000, 1000)
    tensor = (tensor - tensor.mean()) / tensor.std()
    assert torch.isfinite(tensor).all()
    return tensor[None, None].contiguous()


def window_input(window, mean, std):
    tensor = (torch.from_numpy(window.astype(np.float32)) - mean) / std
    assert torch.isfinite(tensor).all()
    return tensor[None, None].contiguous()


def readout(tokens, grid, view, start=None):
    tokens = tokens.reshape(*grid, -1)
    if view["readout"] == "all":
        return tokens.reshape(-1, tokens.shape[-1])
    size = view["readout_tokens"]
    if view["readout"] == "central":
        start = (np.asarray(grid) - size) // 2
    return tokens[start[0]:start[0] + size, start[1]:start[1] + size, start[2]:start[2] + size].reshape(-1, tokens.shape[-1])


def encode(model, x, blocks):
    layers = model.get_intermediate_layers(x.cuda(), n=[block - 1 for block in blocks], return_class_token=True, norm=True)
    return [(patch[0], cls[0]) for patch, cls in layers]


def extract(name, inputs, smoke):
    from .models import load_flexict

    storage, _, _ = context()
    spec = specification()
    model_spec = read_json(PIPELINE)["models"][name]
    request = {"model": name, "model_specification": model_spec, "views": spec["views"], "blocks": spec["blocks"], "layer_rule": spec["layer_rule"], "readout_rule": spec["readout_rule"],
               "inputs": inputs.name, "source": {path: sha256(ROOT / path) for path in SOURCES}, "assets": read_json(ROOT / "configs/model_sources.json")["flexict"],
               "precision": "float32", "tf32": False}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(storage["activations"]) / spec["suite"] / ("smoke" if smoke else "features") / name / key
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", request)
    model, _ = load_flexict(model_spec["checkpoint"])
    model.cuda()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started, checks, seconds, done = time.perf_counter(), [], {view: 0.0 for view in spec["views"]}, 0
    # batches are encoded as their uploads complete; inputs.json arrives last and lists every batch with its SHA-256
    while True:
        pending = [path for path in sorted(inputs.glob("batch_*.h5")) if not (directory / path.name).exists()]
        for path in pending:
            encode_batch(model, name, path, directory / path.name, spec, checks, seconds)
            done += 1
            write_json(directory / "progress.json", {"model": name, "batches_encoded_this_run": done, "seconds": time.perf_counter() - started, "view_seconds": seconds, "updated_at": timestamp()})
            print(f"extract {name}: {path.name} encoded, {done} this run, {(time.perf_counter() - started) / 60:.1f} min, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        if (inputs / "inputs.json").exists():
            expected = [item for item in read_json(inputs / "inputs.json")["batches"] if item["name"].startswith("batch_")]
            if all((directory / item["name"]).exists() for item in expected):
                break
        if not pending:
            time.sleep(60)
    for item in expected:
        with h5py.File(directory / item["name"], "r") as handle:
            assert handle.attrs["source_sha256"] == item["sha256"], item["name"]
    receipt = {"model": name, "directory": str(directory), "batches": len(expected), "inputs_cases_sha256": read_json(inputs / "inputs.json")["cases_sha256"], "seconds": time.perf_counter() - started,
               "view_seconds": seconds, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "official_output_checks": checks, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "completed_at": timestamp(), "smoke": smoke, "test_used": False}
    root = suite_root()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"{name}_{'smoke' if smoke else 'features'}.json", receipt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return directory


def encode_batch(model, name, path, target, spec, checks, seconds):
    values, digest = {}, sha256(path)
    with h5py.File(path, "r") as handle, torch.inference_mode():
        count = len(handle["identifier"])
        for view_name, view in spec["views"].items():
            begun = time.perf_counter()
            # class token and mean patch token of the whole forward (the official window for whole-CT views) give the VLM image embedding through its projection
            outputs, classes, means = ({block: [None] * count for block in spec["blocks"]} for _ in range(3))
            if view["kind"] == "field":
                for index in tqdm(range(count), desc=f"{name} {path.stem} {view_name}", mininterval=10):
                    x = field_input(handle[view_name][index])
                    layers = encode(model, x, spec["blocks"])
                    grid = np.asarray(x.shape[2:]) // PATCH
                    for block, (patch, cls) in zip(spec["blocks"], layers):
                        outputs[block][index] = readout(patch, grid, view).float().cpu().numpy()
                        classes[block][index] = cls.float().cpu().numpy()
                        means[block][index] = patch.mean(0).float().cpu().numpy()
                    if len(checks) < 2 * len(spec["views"]) and index < 2:
                        checks.append(official_check(model, x, layers, spec, view_name))
            else:
                group = handle[view_name]
                readers, owners = {}, {}
                for index, (window, official) in enumerate(zip(group["window_index"][:], group["official_index"][:])):
                    readers.setdefault(int(window), []).append(index)
                    owners.setdefault(int(official), []).append(index)
                for window in tqdm(sorted(set(readers) | set(owners)), desc=f"{name} {path.stem} {view_name}", mininterval=10):
                    x = window_input(group["windows"][window], float(group["mean"][window]), float(group["std"][window]))
                    layers = encode(model, x, spec["blocks"])
                    grid = np.asarray(x.shape[2:]) // PATCH
                    for block, (patch, cls) in zip(spec["blocks"], layers):
                        for index in readers.get(window, []):
                            outputs[block][index] = readout(patch, grid, view, group["readout"][index]).float().cpu().numpy()
                        for index in owners.get(window, []):
                            classes[block][index] = cls.float().cpu().numpy()
                            means[block][index] = patch.mean(0).float().cpu().numpy()
                    if len(checks) < 2 * len(spec["views"]) and window < 2:
                        checks.append(official_check(model, x, layers, spec, view_name))
            for block in spec["blocks"]:
                values[f"{view_name}/block_{block:02d}"] = np.stack(outputs[block])
                values[f"{view_name}/cls_{block:02d}"] = np.stack(classes[block])
                values[f"{view_name}/mean_{block:02d}"] = np.stack(means[block])
            seconds[view_name] += time.perf_counter() - begun
        identifiers = handle["identifier"][:]
    assert all(np.isfinite(value).all() for value in values.values())
    with h5py.File(target.with_suffix(".tmp"), "w") as output:
        output.attrs["source_sha256"] = digest
        output.create_dataset("identifier", data=identifiers)
        for dataset, value in values.items():
            output.create_dataset(dataset, data=value)
    target.with_suffix(".tmp").replace(target)


def official_check(model, x, layers, spec, view_name):
    reference = model(x.cuda(), is_training=True)
    patch, cls = layers[spec["blocks"].index(model.n_blocks)]
    difference = max(float((reference["x_norm_patchtokens"][0] - patch).abs().max()), float((reference["x_norm_clstoken"][0] - cls).abs().max()))
    assert difference < 1e-4, (view_name, difference)
    return {"view": view_name, "max_abs_difference": difference}


# Remote heads.

def load_bags(directory, frames, entries):
    identifiers, pieces = [], []
    for path in sorted(directory.glob("batch_*.h5")):
        with h5py.File(path, "r") as handle:
            identifiers += handle["identifier"][:].astype(str).tolist()
            parts = [handle[dataset][:] for dataset in entries]
            pieces.append(np.concatenate([part[:, None, :] if part.ndim == 2 else part for part in parts], 1))
    values = np.concatenate(pieces)
    position = pd.Series(np.arange(len(identifiers)), index=identifiers)
    assert position.index.is_unique
    return [values[position.loc[frame.identifier].to_numpy()] for frame in frames]


def score(result):
    return result["metrics"]["auroc"], -result["metrics"]["log_loss"]


def fit(name, inputs, smoke):
    from .frozen_heads import fit_spatial

    spec = specification()
    receipt = read_json(suite_root() / f"{name}_{'smoke' if smoke else 'features'}.json")
    directory = Path(receipt["directory"])
    table = pd.read_csv(inputs / "cases.csv", dtype={"PatientID": str})
    frames = [table[table.split == split].sort_values("identifier").reset_index(drop=True) for split in ["train", "development"]]
    assert set(frames[0].PatientID).isdisjoint(frames[1].PatientID)
    root = suite_root() / ("smoke" if smoke else "heads") / name
    epochs = 2 if smoke else spec["epochs"]
    groups = {}

    def run(label, entries):
        bags = load_bags(directory, frames, entries)
        groups[label] = {"entries": entries, "runs": [fit_spatial(name, spec["head"], directory, frames, bags, root / label, index, spec["seed"], lr, epochs=epochs) for index, lr in enumerate(spec["learning_rates"])]}
        del bags
        gc.collect()

    for view in spec["views"]:
        for block in spec["blocks"]:
            run(f"{view}_b{block:02d}", [f"{view}/block_{block:02d}"])
    best_of = lambda labels: max(labels, key=lambda label: max(score(result) for result in groups[label]["runs"]))
    local = best_of([label for label in groups if label.startswith(("f32", "f40"))])
    whole = best_of([label for label in groups if label.startswith(("w15", "w20"))])
    view, block = whole.split("_b")
    run(f"fusion_{local}_{whole}", groups[local]["entries"] + groups[whole]["entries"] + [f"{view}/cls_{block}"])
    best = best_of(list(groups))
    crop = best_of([label for label in groups if label.startswith(("f32", "f40"))])
    context_best = best_of([label for label in groups if not label.startswith(("f32", "f40"))])
    repeated = {}
    for label in dict.fromkeys([best, crop, context_best]):
        chosen = max(groups[label]["runs"], key=score)
        bags = load_bags(directory, frames, groups[label]["entries"])
        repeated[label] = [chosen] + ([] if smoke else [fit_spatial(name, spec["head"], directory, frames, bags, root / label, chosen["candidate"], seed, chosen["lr"], epochs=epochs) for seed in spec["repeat_seeds"]])
        del bags
        gc.collect()
    rows = [{"model": name, "candidate": label, "lr": result["lr"], "seed": result["seed"], "run": str(root / label / result["run_id"]), **{metric: result["metrics"][metric] for metric in METRICS}}
            for label, group in groups.items() for result in group["runs"]]
    rows += [{"model": name, "candidate": label, "lr": result["lr"], "seed": result["seed"], "run": str(root / label / result["run_id"]), **{metric: result["metrics"][metric] for metric in METRICS}}
             for label, results in repeated.items() for result in results[1:]]
    pd.DataFrame(rows).to_csv(root / "candidates.csv", index=False)
    write_json(root / "selection.json", {"model": name, "best": best, "best_crop": crop, "best_context": context_best, "entries": {label: groups[label]["entries"] for label in repeated},
                                         "runs": {label: [str(root / label / result["run_id"]) for result in results] for label, results in repeated.items()}, "completed_at": timestamp(), "smoke": smoke, "test_used": False})
    print(name, "best", best, "crop", crop, "context", context_best, flush=True)


# Report.

def mean_probability(runs):
    frames = [pd.read_csv(Path(run) / "development_predictions.csv", dtype={"PatientID": str}).sort_values("identifier").reset_index(drop=True) for run in runs]
    assert all(frame.identifier.equals(frames[0].identifier) for frame in frames)
    return frames[0].assign(probability=np.mean([frame.probability for frame in frames], axis=0))


def paired_bootstrap(first, second, draws=2000, seed=2025):
    from sklearn.metrics import roc_auc_score

    assert first.identifier.equals(second.identifier)
    patients = first.PatientID.to_numpy()
    groups = pd.Series(np.arange(len(first))).groupby(patients).apply(list).tolist()
    random = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        index = np.concatenate([groups[i] for i in random.integers(0, len(groups), len(groups))])
        labels = first.label.to_numpy()[index]
        if labels.min() == labels.max():
            continue
        differences.append(roc_auc_score(labels, second.probability.to_numpy()[index]) - roc_auc_score(labels, first.probability.to_numpy()[index]))
    point = roc_auc_score(first.label, second.probability) - roc_auc_score(first.label, first.probability)
    return {"difference": point, "ci95": np.percentile(differences, [2.5, 97.5]).tolist(), "draws": len(differences)}


def report(smoke=False):
    from .head import metrics

    spec = specification()
    output = suite_root() / ("smoke_report" if smoke else "report")
    output.mkdir(parents=True, exist_ok=True)
    tables, summary = [], {}
    for name in spec["models"]:
        root = suite_root() / ("smoke" if smoke else "heads") / name
        table = pd.read_csv(root / "candidates.csv")
        selection = read_json(root / "selection.json")
        item = {"best": selection["best"], "best_crop": selection["best_crop"], "best_context": selection["best_context"], "entries": selection["entries"]}
        averaged = {}
        for label, runs in selection["runs"].items():
            averaged[label] = mean_probability(runs)
            repeated = table[table.run.isin(runs)]
            item[label] = {"seeds": len(runs), "mean_metrics": repeated[METRICS].mean().to_dict(), "seed_auroc": repeated.auroc.tolist(), "averaged_probability_metrics": metrics(averaged[label].label, averaged[label].probability)}
        item["context_gain"] = paired_bootstrap(averaged[selection["best_crop"]], averaged[selection["best_context"]])
        summary[name] = item
        tables.append(table)
    table = pd.concat(tables)
    table.to_csv(output / "candidates.csv", index=False)
    write_json(output / "summary.json", {"suite": spec["suite"], "models": summary, "selection": spec["selection"], "source_sha256": {path: sha256(ROOT / path) for path in SOURCES}, "created_at": timestamp(), "smoke": smoke, "test_used": False})
    figure(table, spec, summary, output)
    for name, item in summary.items():
        print(name, "best", item["best"], round(item[item["best"]]["mean_metrics"]["auroc"], 4), "crop", item["best_crop"], round(item[item["best_crop"]]["mean_metrics"]["auroc"], 4),
              "context", item["best_context"], round(item[item["best_context"]]["mean_metrics"]["auroc"], 4), "gain", {key: np.round(value, 4).tolist() for key, value in item["context_gain"].items()}, flush=True)


def figure(table, spec, summary, output):
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": muted})
    views, blocks = list(spec["views"]), spec["blocks"]
    first = table[table.seed == spec["seed"]]
    grid = first[first.candidate.str.match(r"^[a-z]\d+_b\d+$")].groupby(["model", "candidate"]).auroc.max()
    low, high = np.floor(grid.min() * 100) / 100, np.ceil(grid.max() * 100) / 100
    names = {"f32": "32 mm field", "f40": "40 mm field", "c80": "80 mm field,\ncentral 40 mm", "w15": "whole CT 1.5 mm,\nnodule tokens", "w20": "whole CT 2.0 mm,\nnodule tokens"}
    fig, axes = plt.subplots(1, len(spec["models"]), figsize=(6.6 * len(spec["models"]), 4.4), facecolor=surface, squeeze=False)
    for axis, name in zip(axes[0], spec["models"]):
        best = first[first.model == name].groupby("candidate").auroc.max()
        matrix = np.array([[best[f"{view}_b{block:02d}"] for block in blocks] for view in views])
        image = axis.imshow(matrix, cmap="Blues", vmin=low, vmax=high)
        for row in range(len(views)):
            for column in range(len(blocks)):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=surface if matrix[row, column] > (low + high) / 2 else ink, fontsize=8.5)
        axis.set_xticks(range(len(blocks)), [f"block {block}" for block in blocks])
        axis.set_yticks(range(len(views)), [names[view] for view in views])
        item = summary[name]
        fusion = [label for label in best.index if label.startswith("fusion")]
        axis.set_title(f"{name}: best seed-2025 development AUROC over two learning rates, max head\nfusion ({fusion[0].removeprefix('fusion_')}): {best[fusion[0]]:.3f}\n"
                       f"selected {item['best']}: three-seed mean {item[item['best']]['mean_metrics']['auroc']:.3f}", fontsize=8.5, loc="left")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"views_blocks.{suffix}", dpi=170, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "extract", "fit", "report"], required=True)
    parser.add_argument("--model", choices=specification()["models"])
    parser.add_argument("--inputs", type=Path, help="directory holding cases.csv, inputs.json and the batch files")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--smoke", type=int, default=0, help="prepare: scans per split and label; extract, fit and report: use the smoke directories")
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.workers, args.smoke)
    elif args.phase == "extract":
        extract(args.model, args.inputs, bool(args.smoke))
    elif args.phase == "fit":
        fit(args.model, args.inputs, bool(args.smoke))
    else:
        report(bool(args.smoke))


if __name__ == "__main__":
    main()

import argparse
import gc
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .sae_sites import SITES, SpatialEncoder, material, site_affine, volume_with_geometry


def configuration():
    return read_json(ROOT / "configs/sae_preparation.json")


def locations(name):
    storage, _, _ = context()
    suite = configuration()["suite"]
    return Path(storage["activations"]) / suite / name, Path(storage["runs"]) / suite


def stable_seed(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "little")


def check_saved_case(path, name):
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {f"{key}.npy" for key in [*SITES[name], "probability"]}
        for site in SITES[name]:
            with archive.open(f"{site}.npy") as handle:
                assert np.lib.format.read_magic(handle) == (1, 0)
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                assert dtype == np.float32 and len(shape) == 4 and shape[0] == SITES[name][site]["channels"]
                assert not fortran_order
                assert handle.tell() + int(np.prod(shape)) * dtype.itemsize == archive.getinfo(f"{site}.npy").file_size


def extract(name, limit=None):
    directory, root = locations(name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cases").mkdir(exist_ok=True)
    config = configuration()
    source = {"material": material(name), "sites": SITES[name], "source_sha256": sha256(ROOT / "src/sclvmi/sae_sites.py"), "geometry_source_sha256": sha256(ROOT / "src/sclvmi/sae_geometry.py"), "storage_dtype": "float32", "test_used": False}
    source_path = directory / "source.json"
    if source_path.exists():
        assert read_json(source_path) == source, "Changed extraction source requires a new suite"
    else:
        write_json(source_path, source)
    table = load_manifest()
    members = pd.concat([select_cases(table, split, limit) for split in ["train", "development"]]).reset_index(drop=True)
    assert set(members.split) == {"train", "development"}
    assert set(members.query("split == 'train'").PatientID).isdisjoint(members.query("split == 'development'").PatientID)
    cached = {path.stem for path in (directory / "cases").glob("*.json") if path.with_suffix(".npz").exists()}
    cached_count = len(cached.intersection(members.identifier))
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = SpatialEncoder(name)
    started = time.perf_counter()
    new_cases = 0
    for index, row in enumerate(tqdm(members.itertuples(), total=len(members), desc=f"{name} spatial extraction", mininterval=3)):
        target = directory / "cases" / f"{row.identifier}.npz"
        geometry_path = target.with_suffix(".json")
        if target.exists() and geometry_path.exists():
            check_saved_case(target, name)
        else:
            tensor, geometry = volume_with_geometry(row.image, name)
            outputs = encoder.extract(tensor)
            values = {site: outputs[site][0].cpu().numpy() for site in SITES[name]}
            values["probability"] = encoder.probability(outputs["final"])
            for value in values.values():
                assert np.isfinite(value).all()
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, **values)
            temporary.replace(target)
            geometry["sites"] = {site: {**SITES[name][site], "shape": list(values[site].shape), "affine_ras": site_affine(geometry, name, site).tolist()} for site in SITES[name]}
            geometry.update({"identifier": row.identifier, "PatientID": row.PatientID, "split": row.split, "image": row.image, "mask": row.mask})
            write_json(geometry_path, geometry)
            new_cases += 1
            del outputs, values, tensor
        if index % 20 == 0 or index + 1 == len(members):
            write_json(root / "extraction_progress.json", {"stage": "extract", "model": name, "completed": cached_count + new_cases, "validated": index + 1, "total": len(members), "new_cases": new_cases, "elapsed_seconds": time.perf_counter() - started, "updated_at": timestamp()})
    suffix = "smoke" if limit else "full"
    members.to_csv(directory / f"{suffix}_members.csv", index=False)
    write_json(directory / f"{suffix}_extraction.json", {"status": "completed", "records": len(members), "new_cases": new_cases, "seconds": time.perf_counter() - started, "source_sha256": sha256(source_path), "completed_at": timestamp()})
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return directory


def prepare_matrix(name, site, smoke=False):
    directory, root = locations(name)
    config = configuration()
    members = pd.read_csv(directory / ("smoke_members.csv" if smoke else "full_members.csv"), dtype={"PatientID": str})
    destination = directory / (f"{site}_smoke" if smoke else site)
    destination.mkdir(exist_ok=True)
    channels = SITES[name][site]["channels"]
    receipt = destination / "matrix.json"
    if receipt.exists():
        assert read_json(receipt)["members_sha256"] == sha256(directory / ("smoke_members.csv" if smoke else "full_members.csv"))
        return destination
    sizes = {}
    for split in ["train", "development"]:
        selected = members.query("split == @split")
        budget = config["sampling"][f"{split}_tokens_per_patient"]
        patients = sorted(selected.PatientID.unique())
        total = len(patients) * budget
        path = destination / f"{split}.npy"
        complete = destination / f"{split}_matrix.json"
        if complete.exists():
            saved = np.load(path, mmap_mode="r")
            assert saved.shape == (total, channels)
            sizes[split] = {"tokens": total, "patients": len(patients), "cases": len(selected)}
            continue
        mode = "r+" if path.exists() else "w+"
        array = np.lib.format.open_memmap(path, mode=mode, dtype=np.float32, shape=(total, channels))
        marker = destination / f"{split}_resume.json"
        first = read_json(marker)["completed_patients"] if marker.exists() else 0
        counts = []
        for patient_index in tqdm(range(first, len(patients)), desc=f"{name} {site} {split} patients", mininterval=3):
            patient = patients[patient_index]
            rows = selected[selected.PatientID == patient].sort_values("identifier")
            generator = np.random.default_rng(stable_seed(f"{config['sampling']['seed']}:{patient}:{site}"))
            counts = np.bincount(generator.integers(len(rows), size=budget), minlength=len(rows))
            start = patient_index * budget
            for row, count in zip(rows.itertuples(), counts):
                if count == 0:
                    continue
                with np.load(io.BytesIO((directory / "cases" / f"{row.identifier}.npz").read_bytes())) as case:
                    tokens = case[site].reshape(channels, -1).T
                    indices = generator.choice(len(tokens), size=count, replace=count > len(tokens))
                    array[start:start + count] = tokens[indices]
                start += count
            assert start == (patient_index + 1) * budget
            if patient_index % 50 == 0 or patient_index + 1 == len(patients):
                array.flush()
                write_json(marker, {"completed_patients": patient_index + 1, "total": len(patients), "updated_at": timestamp()})
        array.flush()
        del array
        sizes[split] = {"tokens": total, "patients": len(patients), "cases": len(selected)}
        write_json(complete, {"status": "completed", **sizes[split], "completed_at": timestamp()})
    training = np.load(destination / "train.npy", mmap_mode="r")
    sum_x = np.zeros(channels, dtype=np.float64)
    sum_square = np.zeros(channels, dtype=np.float64)
    for start in range(0, len(training), 4096):
        values = np.asarray(training[start:start + 4096], dtype=np.float64)
        assert np.isfinite(values).all()
        sum_x += values.sum(0)
        sum_square += np.square(values).sum(0)
    mean = sum_x / len(training)
    variance = np.maximum(sum_square / len(training) - np.square(mean), 0)
    scale = float(np.sqrt(variance.mean()))
    assert scale > 0
    np.savez(destination / "normalization.npz", mean=mean.astype(np.float32), scale=np.float32(scale), variance=variance)
    write_json(receipt, {"status": "completed", "model": name, "site": site, "channels": channels, "splits": sizes, "normalization": config["normalization"], "members_sha256": sha256(directory / ("smoke_members.csv" if smoke else "full_members.csv")), "source_sha256": sha256(directory / "source.json"), "completed_at": timestamp()})
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vista", "fmcib"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--matrix-only", action="store_true")
    args = parser.parse_args()
    if not args.matrix_only:
        extract(args.model, args.limit)
    for site in SITES[args.model]:
        prepare_matrix(args.model, site, smoke=bool(args.limit))


if __name__ == "__main__":
    main()

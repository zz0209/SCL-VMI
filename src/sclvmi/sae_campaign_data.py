import argparse
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae_campaign import campaign_root, configuration, digest, ensure_identity, pause_requested, progress
from .sae_sites import SITES, SpatialEncoder


def check_pause(root, worker, task, completed, total, started):
    if pause_requested(root, worker):
        progress(root, task, "paused", completed, total, started, worker=worker)
        raise SystemExit(75)


def prepare_additional(root, name, limit=None, worker="local"):
    config = configuration()
    storage, _, _ = context()
    old = Path(storage["activations"]) / config["previous_suite"] / name
    destination = root / "cases" / name
    destination.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv(old / "full_members.csv", dtype={"PatientID": str})
    assert set(members.split) == {"train", "development"}
    assert set(members.query("split == 'train'").PatientID).isdisjoint(members.query("split == 'development'").PatientID)
    if limit:
        members = members.groupby(["split", "label"], sort=True).head(limit)
    ensure_identity(destination / "source.json", {
        "previous_source_sha256": sha256(old / "source.json"),
        "members_sha256": sha256(old / "full_members.csv"),
        "sites": config["models"][name], "code_sha256": sha256(Path(__file__)),
        "test_used": False,
    })
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = SpatialEncoder(name)
    base_site = "layer2" if name == "fmcib" else "stage3"
    additions = {key: spec for key, spec in config["models"][name].items() if key not in SITES[name] and spec["kind"] == "spatial"}
    started = time.monotonic()
    task = f"extract-{name}"
    for index, row in enumerate(members.itertuples()):
        target = destination / f"{row.identifier}.npz"
        if not target.exists():
            check_pause(root, worker, task, index, len(members), started)
            with np.load(old / "cases" / f"{row.identifier}.npz") as saved:
                base = torch.from_numpy(saved[base_site]).cuda()[None]
                probability = saved["probability"]
            captures = {}
            handles = []
            for site, spec in additions.items():
                def capture(module, inputs, output, key=site):
                    captures[key] = output.detach().clone()
                handles.append(encoder.model.get_submodule(spec["module"]).register_forward_hook(capture))
            final = encoder.continue_from(base_site, base)
            for handle in handles:
                handle.remove()
            continued_probability = encoder.probability(final)
            np.testing.assert_allclose(continued_probability, probability, rtol=0, atol=1e-5)
            values = {site: value[0].cpu().numpy() for site, value in captures.items()}
            values["global"] = final[0].mean((1, 2, 3), keepdim=True).cpu().numpy()
            values["probability"] = probability
            for site, spec in config["models"][name].items():
                if site in values:
                    assert values[site].shape[0] == spec["channels"] and np.isfinite(values[site]).all()
            temporary = target.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, **values)
            temporary.replace(target)
            del base, final, captures, values
        if index % 20 == 0 or index + 1 == len(members):
            progress(root, task, "completed" if index + 1 == len(members) else "running", index + 1, len(members), started, worker=worker)
    members.to_csv(destination / ("smoke_members.csv" if limit else "members.csv"), index=False)
    write_json(destination / ("smoke_extraction.json" if limit else "extraction.json"), {"state": "completed", "records": len(members), "elapsed_seconds": time.monotonic() - started, "completed_at": timestamp()})
    del encoder
    gc.collect()
    torch.cuda.empty_cache()


def prepare_matrices(root, name, smoke=False, worker="local"):
    config = configuration()
    storage, _, _ = context()
    old = Path(storage["activations"]) / config["previous_suite"] / name
    additional = root / "cases" / name
    member_path = additional / ("smoke_members.csv" if smoke else "members.csv")
    members = pd.read_csv(member_path, dtype={"PatientID": str})
    sites = config["models"][name]
    first = members.iloc[0].identifier
    shapes = {}
    with np.load(old / "cases" / f"{first}.npz") as legacy, np.load(additional / f"{first}.npz") as extra:
        for site in sites:
            shapes[site] = (legacy if site in SITES[name] else extra)[site].shape
    matrix_base = root / ("smoke_matrices" if smoke else "matrices") / name
    matrix_base.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for split in ["train", "development"]:
        selected = members.query("split == @split").sort_values(["PatientID", "identifier"])
        groups = list(selected.groupby("PatientID", sort=True))
        budget = config["sampling"][f"{split}_tokens_per_patient"]
        arrays = {}
        offsets = {}
        for site in sites:
            directory = matrix_base / site
            directory.mkdir(exist_ok=True)
            count_per_case = int(np.prod(shapes[site][1:]))
            lengths = [min(budget, len(frame) * count_per_case) for _, frame in groups]
            offsets[site] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
            path = directory / f"{split}.npy"
            arrays[site] = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=np.float32, shape=(int(offsets[site][-1]), sites[site]["channels"]))
            offset_path = directory / f"{split}_offsets.npy"
            if offset_path.exists():
                np.testing.assert_array_equal(np.load(offset_path), offsets[site])
            else:
                np.save(offset_path, offsets[site])
        marker = matrix_base / f"{split}_resume.json"
        first_patient = read_json(marker)["completed"] if marker.exists() else 0
        task = f"matrix-{name}-{split}"
        for patient_index in range(first_patient, len(groups)):
            check_pause(root, worker, task, patient_index, len(groups), started)
            patient, rows = groups[patient_index]
            sampled = {}
            for site in sites:
                generator = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{config['sampling']['seed']}:{name}:{site}:{patient}".encode()).digest()[:8], "little"))
                positions = int(np.prod(shapes[site][1:]))
                chosen = generator.choice(len(rows) * positions, size=int(offsets[site][patient_index + 1] - offsets[site][patient_index]), replace=False)
                sampled[site] = chosen
            for case_index, row in enumerate(rows.itertuples()):
                with np.load(old / "cases" / f"{row.identifier}.npz") as legacy, np.load(additional / f"{row.identifier}.npz") as extra:
                    for site, spec in sites.items():
                        positions = int(np.prod(shapes[site][1:]))
                        chosen = sampled[site]
                        selected_indices = np.flatnonzero(chosen // positions == case_index)
                        if len(selected_indices) == 0:
                            continue
                        values = (legacy if site in SITES[name] else extra)[site]
                        assert tuple(values.shape) == tuple(shapes[site])
                        tokens = values.reshape(spec["channels"], -1).T
                        arrays[site][offsets[site][patient_index] + selected_indices] = tokens[chosen[selected_indices] % positions]
            if patient_index % 20 == 0 or patient_index + 1 == len(groups):
                for array in arrays.values():
                    array.flush()
                write_json(marker, {"completed": patient_index + 1, "total": len(groups), "updated_at": timestamp()})
                progress(root, task, "completed" if patient_index + 1 == len(groups) else "running", patient_index + 1, len(groups), started, worker=worker)
        for array in arrays.values():
            array.flush()
        arrays.clear()
    for site, spec in sites.items():
        destination = matrix_base / site
        if (destination / "matrix.json").exists():
            continue
        train = np.load(destination / "train.npy", mmap_mode="r")
        train_offsets = np.load(destination / "train_offsets.npy")
        mean = np.zeros(spec["channels"], dtype=np.float64)
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
        metadata = {
            "model": name, "site": site, "channels": spec["channels"], "native_shape": list(shapes[site][1:]),
            "kind": spec["kind"], "members_sha256": sha256(member_path),
            "sampling": config["sampling"], "test_used": False,
            "normalization": "equal training-patient channel mean and global RMS",
            "splits": {}, "files": {},
        }
        for split in ["train", "development"]:
            offsets = np.load(destination / f"{split}_offsets.npy")
            metadata["splits"][split] = {"patients": len(offsets) - 1, "tokens": int(offsets[-1]), "cases": int((members.split == split).sum()), "unique_positions": int(offsets[-1])}
        for path in sorted(destination.glob("*.np*")):
            progress(root, f"hash-{name}-{site}", "running", len(metadata["files"]), 5, started, file=path.name)
            metadata["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        metadata["identity"] = digest(metadata)
        metadata["created_at"] = timestamp()
        write_json(destination / "matrix.json", metadata)
        progress(root, f"hash-{name}-{site}", "completed", 5, 5, started)
        print("MATRIX_READY", name, site, metadata["splits"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--model", choices=["fmcib", "vista"], required=True)
    parser.add_argument("--stage", choices=["all", "extract", "matrix"], default="all")
    parser.add_argument("--smoke", type=int)
    parser.add_argument("--worker", default="local")
    args = parser.parse_args()
    root = campaign_root(args.root)
    if args.stage in {"all", "extract"}:
        prepare_additional(root, args.model, args.smoke, args.worker)
    if args.stage in {"all", "matrix"}:
        prepare_matrices(root, args.model, bool(args.smoke), args.worker)


if __name__ == "__main__":
    main()

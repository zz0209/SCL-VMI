import argparse
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch

from .context import context, read_json, sha256, timestamp, write_json
from .sae_campaign import campaign_root, configuration, ensure_identity, pause_requested, progress
from .sae_sites import material, volume_with_geometry


def prepare_inputs(root, model, smoke=0):
    storage, _, _ = context()
    source = Path(storage["runs"]) / "20261009_feature_exploration/members.csv"
    members = pd.read_csv(source, dtype={"PatientID": str})
    full_path = Path(storage["activations"]) / configuration()["previous_suite"] / model / "full_members.csv"
    full = pd.read_csv(full_path, dtype={"PatientID": str})
    chosen = full.loc[full.identifier.isin(members.identifier)].sort_values("identifier")
    assert len(chosen) == len(members) == chosen.PatientID.nunique()
    assert set(chosen.split) == {"development"}
    if smoke:
        chosen = chosen.head(smoke)
    destination = root / ("viewer_smoke" if smoke else "viewer") / model
    destination.mkdir(parents=True, exist_ok=True)
    ensure_identity(destination / "request.json", {"member_source_sha256": sha256(source), "model": model, "identifiers": chosen.identifier.tolist(), "source_sha256": sha256(Path(__file__)), "test_used": False})
    settings = material(model)["preprocessing"]["settings"]
    started = time.monotonic()
    records = []
    for index, row in enumerate(chosen.itertuples()):
        if pause_requested(root, "local"):
            progress(root, f"viewer-input-{model}", "paused", index, len(chosen), started, worker="local")
            raise SystemExit(75)
        path = destination / f"case_{index:03d}.h5"
        if not path.exists():
            tensor, geometry = volume_with_geometry(row.image, model)
            affine = np.asarray(geometry["input_affine_ras"])
            orientation = nib.orientations.io_orientation(affine)
            transform = nib.orientations.ornt_transform(orientation, nib.orientations.axcodes2ornt(("R", "A", "S")))
            input_values = tensor.numpy()[0]
            canonical_affine = affine @ nib.orientations.inv_ornt_aff(transform, input_values.shape)
            low, high = (-1024, 2048) if model == "fmcib" else settings["intensity"]
            hu = nib.orientations.apply_orientation(input_values * (high - low) + low, transform)
            temporary = path.with_suffix(".tmp.h5")
            with h5py.File(temporary, "w") as saved:
                saved.create_dataset("input_hu", data=hu.astype(np.float32), compression="gzip", compression_opts=1)
                saved.attrs["input_affine_ras"] = affine
                saved.attrs["canonical_affine_ras"] = canonical_affine
                saved.attrs["identifier"] = row.identifier
                saved.attrs["model"] = model
            temporary.replace(path)
        with h5py.File(path, "r") as saved:
            assert saved.attrs["identifier"] == row.identifier
            records.append({"index": index, "label": f"C{index + 1:03d}", "identifier": row.identifier, "split": "development", "shape": list(saved["input_hu"].shape), "spacing": nib.affines.voxel_sizes(saved.attrs["canonical_affine_ras"]).tolist()})
        progress(root, f"viewer-input-{model}", "running", index + 1, len(chosen), started, worker="local")
    write_json(destination / "cases.json", {"cases": records, "created_at": timestamp(), "test_used": False})
    progress(root, f"viewer-input-{model}", "completed", len(chosen), len(chosen), started, worker="local")


def publish_index(root):
    selection = read_json(root / "selection.json")
    assert selection["state"] == "completed"
    entries = []
    for selected in selection["selected"]:
        for run_id in selected["replicas"]:
            run = root / "runs" / run_id
            result = read_json(run / "result.json")
            downstream = read_json(run / "downstream.json")
            assert result["feature_checks_pass"] and downstream["checks_pass"]
            assert sha256(run / "dictionary.pt") == result["dictionary_sha256"] == downstream["dictionary_sha256"]
            request = result["request"]
            assert (root / "viewer" / request["model"] / "cases.json").exists()
            entries.append({"run_id": run_id, "model": request["model"], "site": request["site"], "seed": request["seed"], "features": request["dict_size"], "k": request["k"], "role": selected["role"], "dictionary_sha256": result["dictionary_sha256"], "development": result["development"], "downstream": downstream["metrics"], "site_specification": request["site_specification"]})
    result = {"state": "completed", "dictionaries": entries, "created_at": timestamp(), "test_used": False}
    write_json(root / "viewer" / "index.json", result)
    write_json(root / "viewer_export.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fmcib", "vista"])
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()
    root = campaign_root()
    torch.set_num_threads(3)
    if args.model:
        prepare_inputs(root, args.model, args.smoke)
    if args.publish:
        print(publish_index(root), flush=True)

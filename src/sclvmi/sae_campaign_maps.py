import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .context import context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_campaign import campaign_root, configuration, ensure_identity, pause_requested, progress
from .sae_sites import SITES


def prepare_responses(root, run_id):
    run = root / "runs" / run_id
    request = read_json(run / "request.json")
    cases_path = root / "viewer" / request["model"] / "cases.json"
    cases = read_json(cases_path)["cases"]
    destination = root / "viewer/responses" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    ensure_identity(destination / "request.json", {"dictionary_sha256": sha256(run / "dictionary.pt"), "cases_sha256": sha256(cases_path), "code_sha256": sha256(Path(__file__)), "test_used": False})
    receipt = destination / "result.json"
    if receipt.exists():
        return read_json(receipt)
    storage, _, _ = context()
    model, site = request["model"], request["site"]
    source = Path(storage["activations"]) / configuration()["previous_suite"] / model / "cases" if site in SITES[model] else root / "cases" / model
    dictionary = SpatialDictionary(run / "dictionary.pt")
    path = destination / "responses.h5"
    started = time.monotonic()
    with h5py.File(path, "a") as saved:
        if "completed" not in saved:
            for key in ["maximum", "mean", "active_fraction"]:
                saved.create_dataset(key, shape=(len(cases), request["dict_size"]), dtype="f4", chunks=(1, request["dict_size"]), compression="gzip", compression_opts=1)
            saved.create_dataset("completed", data=np.zeros(len(cases), dtype=bool))
        for row in cases:
            index = row["index"]
            if saved["completed"][index]:
                continue
            if pause_requested(root, "local"):
                progress(root, f"viewer-responses-{run_id}", "paused", int(saved["completed"][...].sum()), len(cases), started, worker="local")
                raise SystemExit(75)
            with np.load(source / f"{row['identifier']}.npz") as value:
                native = torch.from_numpy(value[site]).cuda()
            with torch.inference_mode():
                encoded = dictionary.spatial_activations(native).flatten(1)
            saved["maximum"][index] = encoded.amax(1).cpu().numpy()
            saved["mean"][index] = encoded.mean(1).cpu().numpy()
            saved["active_fraction"][index] = (encoded > 0).float().mean(1).cpu().numpy()
            saved.flush()
            saved["completed"][index] = True
            saved.flush()
            if (index + 1) % 8 == 0 or index + 1 == len(cases):
                progress(root, f"viewer-responses-{run_id}", "running", index + 1, len(cases), started, worker="local")
        assert saved["completed"][...].all()
        assert all(np.isfinite(saved[key][...]).all() for key in ["maximum", "mean", "active_fraction"])
    result = {"state": "completed", "run_id": run_id, "cases": len(cases), "features": request["dict_size"], "dictionary_sha256": sha256(run / "dictionary.pt"), "test_used": False, "completed_at": timestamp()}
    write_json(receipt, result)
    progress(root, f"viewer-responses-{run_id}", "completed", len(cases), len(cases), started, worker="local")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    for run_id in args.runs:
        print(prepare_responses(campaign_root(), run_id), flush=True)

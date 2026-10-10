import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .context import context, read_json, sha256, timestamp, write_json
from .models import spatial_features
from .sae import SpatialDictionary
from .sae_campaign import campaign_root, configuration, ensure_identity, pause_requested, progress
from .sae_sites import SITES, SpatialEncoder, volume_with_geometry


def metrics(frame):
    lengths = frame.groupby("PatientID").identifier.transform("size").to_numpy()
    weights = 1.0 / lengths
    original, recovered, labels = frame.original.to_numpy(), frame.recovered.to_numpy(), frame.label.to_numpy()
    return {
        "cases": len(frame), "patients": frame.PatientID.nunique(),
        "probability_mae": float(np.average(np.abs(original - recovered), weights=weights)),
        "probability_rmse": float(np.sqrt(np.average((original - recovered) ** 2, weights=weights))),
        "original_auroc": float(roc_auc_score(labels, original, sample_weight=weights)),
        "recovered_auroc": float(roc_auc_score(labels, recovered, sample_weight=weights)),
        "original_ap": float(average_precision_score(labels, original, sample_weight=weights)),
        "recovered_ap": float(average_precision_score(labels, recovered, sample_weight=weights)),
        "weighting": "equal patient weight, equal case weight within patient",
    }


def evaluate(root, run_id, smoke=False):
    config = configuration()
    run = root / "runs" / run_id
    request = read_json(run / "request.json")
    name, site = request["model"], request["site"]
    storage, _, _ = context()
    old = Path(storage["activations"]) / config["previous_suite"] / name
    members_path = root / "cases" / name / "smoke_members.csv" if smoke else old / "full_members.csv"
    members = pd.read_csv(members_path, dtype={"PatientID": str}).query("split == 'development'").sort_values("identifier")
    if smoke:
        members = members.groupby("label", sort=True).head(2)
    destination = run / ("downstream_smoke" if smoke else "downstream_full")
    destination.mkdir(exist_ok=True)
    identity = {"dictionary_sha256": sha256(run / "dictionary.pt"), "code_sha256": sha256(Path(__file__)), "members": members.identifier.tolist(), "model_material": name, "test_used": False}
    ensure_identity(destination / "request.json", identity)
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dictionary = SpatialDictionary(run / "dictionary.pt")
    encoder = SpatialEncoder(name)
    pooled = site == "global"
    final_site = "layer4" if name == "fmcib" else "stage4"
    base = root / "cases" / name
    predictions_path = destination / "predictions.csv"
    rows = pd.read_csv(predictions_path, dtype={"PatientID": str}).to_dict("records") if predictions_path.exists() else []
    done = {row["identifier"] for row in rows}
    started = time.monotonic()
    task = f"evaluate-{run_id}"
    for row in members.itertuples():
        if row.identifier in done:
            continue
        if pause_requested(root, "local"):
            pd.DataFrame(rows).to_csv(predictions_path, index=False)
            progress(root, task, "paused", len(rows), len(members), started, worker="local")
            raise SystemExit(75)
        source = old / "cases" if site in SITES[name] else base
        with np.load(source / f"{row.identifier}.npz") as saved:
            feature = torch.from_numpy(saved[site]).cuda()
            original = float(saved["probability"][0])
            final = torch.from_numpy(saved[final_site]).cuda()[None] if pooled else None
        channels, *shape = feature.shape
        recovered_map = dictionary.reconstruct(feature.reshape(channels, -1).T).T.reshape(1, channels, *shape)
        if pooled:
            if name == "vista":
                final = encoder.continue_from(final_site, final)
            continued = final
            recovered_final = final - feature[None] + recovered_map
        else:
            continued = encoder.continue_from(site, feature[None])
            recovered_final = encoder.continue_from(site, recovered_map)
        continued_probability = float(encoder.probability(continued)[0])
        assert abs(continued_probability - original) < 1e-5
        recovered = float(encoder.probability(recovered_final)[0])
        rows.append({"identifier": row.identifier, "PatientID": row.PatientID, "label": row.label, "original": original, "recovered": recovered, "identity_error": abs(continued_probability - original)})
        if len(rows) % 25 == 0 or len(rows) == len(members):
            pd.DataFrame(rows).to_csv(predictions_path, index=False)
            progress(root, task, "running", len(rows), len(members), started, worker="local")
    frame = pd.DataFrame(rows)
    frame.to_csv(predictions_path, index=False)
    assert frame.identifier.is_unique and set(frame.identifier) == set(members.identifier)
    raw_checks = []
    for row in members.groupby("label", sort=True).head(1).itertuples():
        tensor, _ = volume_with_geometry(row.image, name)
        module_name = config["models"][name][final_site if pooled else site]["module"]
        captured = {}
        handle = encoder.model.get_submodule(module_name).register_forward_hook(lambda module, inputs, output: captured.update(value=output.detach().clone()))
        with torch.inference_mode():
            final = spatial_features(encoder.model, name, tensor[None].cuda())
        handle.remove()
        if pooled:
            mean = final.mean((2, 3, 4), keepdim=True)
            replacement = final - mean + dictionary.reconstruct(mean.flatten(2).transpose(1, 2)[0]).T.reshape_as(mean)
            probability = float(encoder.probability(replacement)[0])
        else:
            feature = captured["value"]
            replacement = dictionary.reconstruct(feature[0].flatten(1).T).T.reshape_as(feature)
            handle = encoder.model.get_submodule(module_name).register_forward_hook(lambda module, inputs, output: replacement.to(output))
            with torch.inference_mode():
                hooked = spatial_features(encoder.model, name, tensor[None].cuda())
            handle.remove()
            torch.testing.assert_close(hooked, encoder.continue_from(site, replacement), rtol=0, atol=0)
            probability = float(encoder.probability(hooked)[0])
        expected = frame.loc[frame.identifier == row.identifier, "recovered"].item()
        assert abs(probability - expected) < 1e-5
        raw_checks.append({"identifier": row.identifier, "absolute_probability_error": abs(probability - expected)})
    values = metrics(frame)
    limits = config["acceptance"]
    checks = {"probability_mae": values["probability_mae"] <= limits["max_probability_mae"], "auroc": values["original_auroc"] - values["recovered_auroc"] <= limits["max_auroc_decrease"], "ap": values["original_ap"] - values["recovered_ap"] <= limits["max_ap_decrease"]}
    result = {"state": "completed", "run_id": run_id, "metrics": values, "checks": checks, "checks_pass": all(checks.values()), "raw_checks": raw_checks, "completed_at": timestamp(), "test_used": False, "dictionary_sha256": identity["dictionary_sha256"]}
    write_json(destination / "result.json", result)
    if not smoke:
        write_json(run / "downstream.json", result)
    progress(root, task, "completed", len(rows), len(members), started, worker="local")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(evaluate(campaign_root(args.root), args.run_id, args.smoke), flush=True)

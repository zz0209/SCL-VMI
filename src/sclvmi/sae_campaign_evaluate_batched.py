import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .context import context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_campaign import campaign_root, configuration, ensure_identity, pause_requested, progress
from .sae_campaign_evaluate import metrics
from .sae_predict import SpatialSAEPipeline
from .sae_sites import SITES, SpatialEncoder


@torch.inference_mode()
def predict_batch(encoder, dictionary, source, rows):
    request = dictionary.request
    model, site = request["model"], request["site"]
    final_site = "layer4" if model == "fmcib" else "stage4"
    originals, recovered, probabilities = [], [], []
    for row in rows:
        with np.load(source / f"{row.identifier}.npz") as saved:
            feature = torch.from_numpy(saved[site]).to(encoder.device)
            probabilities.append(float(saved["probability"][0]))
            final = torch.from_numpy(saved[final_site]).to(encoder.device) if site == "global" else feature
        reconstructed = dictionary.reconstruct(feature.flatten(1).T).T.reshape_as(feature)
        if site == "global":
            if model == "vista":
                final = encoder.continue_from(final_site, final[None])[0]
            originals.append(final)
            recovered.append(final - feature + reconstructed)
        else:
            originals.append(feature)
            recovered.append(reconstructed)
    original_batch, recovered_batch = torch.stack(originals), torch.stack(recovered)
    if site != "global":
        original_batch = encoder.continue_from(site, original_batch)
        recovered_batch = encoder.continue_from(site, recovered_batch)
    original_values = encoder.probability(original_batch)
    recovered_values = encoder.probability(recovered_batch)
    np.testing.assert_allclose(original_values, probabilities, rtol=0, atol=1e-5)
    return [{"identifier": row.identifier, "PatientID": row.PatientID, "label": row.label, "original": original, "recovered": float(value), "identity_error": abs(float(actual) - original)} for row, original, value, actual in zip(rows, probabilities, recovered_values, original_values)]


def evaluate(root, run_id, batch_size=8, witness=False):
    run = root / "runs" / run_id
    request = read_json(run / "request.json")
    model, site = request["model"], request["site"]
    storage, _, _ = context()
    old = Path(storage["activations"]) / configuration()["previous_suite"] / model
    members = pd.read_csv(old / "full_members.csv", dtype={"PatientID": str}).query("split == 'development'").sort_values("identifier")
    if witness:
        members = members.groupby("label", sort=True).head(4)
    source = old / "cases" if site in SITES[model] else root / "cases" / model
    destination = run / ("downstream_batch_witness" if witness else "downstream_batched")
    destination.mkdir(exist_ok=True)
    identity = {"dictionary_sha256": sha256(run / "dictionary.pt"), "code_sha256": sha256(Path(__file__)), "members": members.identifier.tolist(), "batch_size": batch_size, "test_used": False}
    ensure_identity(destination / "request.json", identity)
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    assert witness or not (run / "downstream.json").exists()
    dictionary = SpatialDictionary(run / "dictionary.pt")
    encoder = SpatialEncoder(model)
    predictions_path = destination / "predictions.csv"
    records = pd.read_csv(predictions_path, dtype={"PatientID": str}).to_dict("records") if predictions_path.exists() else []
    done = {row["identifier"] for row in records}
    remaining = list(members.loc[~members.identifier.isin(done)].itertuples())
    started = time.monotonic()
    task = f"batch-witness-{run_id}" if witness else f"evaluate-{run_id}"
    for start in range(0, len(remaining), batch_size):
        if pause_requested(root, "local"):
            progress(root, task, "paused", len(records), len(members), started, worker="local")
            raise SystemExit(75)
        batch = remaining[start:start + batch_size]
        predicted = predict_batch(encoder, dictionary, source, batch)
        if witness:
            individual = [predict_batch(encoder, dictionary, source, [row])[0]["recovered"] for row in batch]
            np.testing.assert_allclose([row["recovered"] for row in predicted], individual, rtol=0, atol=1e-5)
        records.extend(predicted)
        temporary = predictions_path.with_suffix(".tmp.csv")
        pd.DataFrame(records).to_csv(temporary, index=False)
        temporary.replace(predictions_path)
        progress(root, task, "running", len(records), len(members), started, worker="local")
    frame = pd.DataFrame(records)
    assert frame.identifier.is_unique and set(frame.identifier) == set(members.identifier)
    if witness:
        baseline_path = run / "downstream_full/predictions.csv"
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path).set_index("identifier")
            np.testing.assert_allclose(frame.recovered, baseline.loc[frame.identifier, "recovered"], rtol=0, atol=1e-5)
        result = {"state": "passed", "run_id": run_id, "batch_size": batch_size, "cases": len(frame), "single_case_agreement_atol": 1e-5, "maximum_original_probability_error": float(frame.identity_error.max()), "elapsed_seconds": time.monotonic() - started, "created_at": timestamp()}
    else:
        pipeline = SpatialSAEPipeline(run / "dictionary.pt", encoder=encoder)
        raw_checks = []
        for row in members.groupby("label", sort=True).head(1).itertuples():
            bundle = pipeline.extract(row.image)
            expected = frame.loc[frame.identifier == row.identifier, "recovered"].item()
            error = abs(bundle["geometry"]["reconstructed_probability"] - expected)
            assert error < 1e-5
            raw_checks.append({"identifier": row.identifier, "absolute_probability_error": error})
        values = metrics(frame)
        limits = configuration()["acceptance"]
        checks = {"probability_mae": values["probability_mae"] <= limits["max_probability_mae"], "auroc": values["original_auroc"] - values["recovered_auroc"] <= limits["max_auroc_decrease"], "ap": values["original_ap"] - values["recovered_ap"] <= limits["max_ap_decrease"]}
        result = {"state": "completed", "run_id": run_id, "metrics": values, "checks": checks, "checks_pass": all(checks.values()), "raw_checks": raw_checks, "completed_at": timestamp(), "test_used": False, "dictionary_sha256": identity["dictionary_sha256"], "batch_size": batch_size, "elapsed_seconds": time.monotonic() - started}
        write_json(run / "downstream.json", result)
    write_json(destination / "result.json", result)
    progress(root, task, "completed", len(records), len(members), started, worker="local")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--witness", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print(evaluate(campaign_root(args.root), args.run_id, args.batch_size, args.witness), flush=True)

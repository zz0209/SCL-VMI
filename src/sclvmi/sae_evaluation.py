import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_data import configuration, locations
from .sae_sites import SpatialEncoder, volume_with_geometry


def prediction_metrics(frame):
    original = frame.original.to_numpy()
    recovered = frame.recovered.to_numpy()
    return {
        "cases": len(frame), "patients": frame.PatientID.nunique(),
        "probability_mae": float(np.mean(np.abs(recovered - original))),
        "probability_rmse": float(np.sqrt(np.mean((recovered - original) ** 2))),
        "probability_error_quantiles": np.quantile(np.abs(recovered - original), [0.5, 0.9, 0.99, 1]).tolist(),
        "original_auroc": float(roc_auc_score(frame.label, original)),
        "recovered_auroc": float(roc_auc_score(frame.label, recovered)),
        "original_ap": float(average_precision_score(frame.label, original)),
        "recovered_ap": float(average_precision_score(frame.label, recovered)),
    }


def evaluate_downstream(run_id, patient_limit=None, smoke=False):
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = configuration()
    storage, _, _ = context()
    run = Path(storage["runs"]) / config["suite"] / run_id
    request = read_json(run / "request.json")
    name, site = request["model"], request["site"]
    directory, _ = locations(name)
    members = pd.read_csv(directory / ("smoke_members.csv" if smoke else "full_members.csv"), dtype={"PatientID": str})
    members = members.query("split == 'development'").sort_values("identifier")
    if patient_limit:
        patients = np.asarray(sorted(members.PatientID.unique()))
        chosen = np.random.default_rng(20261008).choice(patients, size=min(patient_limit, len(patients)), replace=False)
        members = members[members.PatientID.isin(chosen)]
    assert members.label.nunique() == 2
    suffix = "smoke" if smoke else f"patients{patient_limit}" if patient_limit else "full"
    destination = run / f"downstream_{suffix}"
    destination.mkdir(exist_ok=True)
    identity = {"dictionary_sha256": sha256(run / "dictionary.pt"), "source_sha256": sha256(Path(__file__)), "identifiers": members.identifier.tolist(), "test_used": False}
    if (destination / "request.json").exists():
        assert read_json(destination / "request.json") == identity
    else:
        write_json(destination / "request.json", identity)
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    dictionary = SpatialDictionary(run / "dictionary.pt")
    encoder = SpatialEncoder(name)
    prediction_path = destination / "predictions.csv"
    rows = pd.read_csv(prediction_path, dtype={"PatientID": str}).to_dict("records") if prediction_path.exists() else []
    done = {row["identifier"] for row in rows}
    for row in tqdm(members.itertuples(), total=len(members), desc=f"{run_id} classifier", mininterval=3):
        if row.identifier in done:
            continue
        with np.load(directory / "cases" / f"{row.identifier}.npz") as case:
            feature_map = torch.from_numpy(case[site]).cuda()
            original = float(case["probability"][0])
        channels, *shape = feature_map.shape
        tokens = feature_map.reshape(channels, -1).T
        reconstruction = dictionary.reconstruct(tokens).T.reshape(1, channels, *shape)
        recovered = float(encoder.probability(encoder.continue_from(site, reconstruction))[0])
        continued = float(encoder.probability(encoder.continue_from(site, feature_map[None]))[0])
        assert abs(continued - original) < 1e-5
        rows.append({"identifier": row.identifier, "PatientID": row.PatientID, "label": row.label, "original": original, "recovered": recovered, "identity_error": abs(continued - original)})
        if len(rows) % 25 == 0:
            pd.DataFrame(rows).to_csv(prediction_path, index=False)
            write_json(destination / "progress.json", {"completed": len(rows), "total": len(members), "updated_at": timestamp()})
    predictions = pd.DataFrame(rows)
    predictions.to_csv(prediction_path, index=False)
    assert predictions.identifier.is_unique and set(predictions.identifier) == set(members.identifier)
    raw_checks = []
    for row in members.groupby("label", sort=True).head(2).itertuples():
        tensor, _ = volume_with_geometry(row.image, name)
        extracted = encoder.extract(tensor, sites=[site])
        with np.load(directory / "cases" / f"{row.identifier}.npz") as saved:
            torch.testing.assert_close(extracted[site][0].cpu(), torch.from_numpy(saved[site]), rtol=0, atol=0)
        features = extracted[site]
        channels = features.shape[1]
        recovered_map = dictionary.reconstruct(features[0].reshape(channels, -1).T).T.reshape_as(features)
        hooked = encoder.replace(tensor, site, recovered_map)
        continued = encoder.continue_from(site, recovered_map)
        torch.testing.assert_close(hooked, continued, rtol=0, atol=0)
        probability = float(encoder.probability(hooked)[0])
        expected = predictions.loc[predictions.identifier == row.identifier, "recovered"].item()
        assert abs(probability - expected) < 1e-5
        raw_checks.append({"identifier": row.identifier, "label": row.label, "probability_error": abs(probability - expected)})
    metrics = prediction_metrics(predictions)
    limits = config["acceptance"]
    checks = {"probability_mae": metrics["probability_mae"] <= limits["max_probability_mae"], "auroc": metrics["original_auroc"] - metrics["recovered_auroc"] <= limits["max_auroc_decrease"], "ap": metrics["original_ap"] - metrics["recovered_ap"] <= limits["max_ap_decrease"]}
    result = {"status": "completed", "run_id": run_id, "scope": suffix, "metrics": metrics, "checks": checks, "checks_pass": all(checks.values()), "raw_image_checks": raw_checks, "dictionary_sha256": identity["dictionary_sha256"], "completed_at": timestamp(), "test_used": False}
    write_json(destination / "result.json", result)
    print("DOWNSTREAM_COMPLETED", run_id, metrics, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--patients", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    evaluate_downstream(args.run_id, args.patients, args.smoke)


if __name__ == "__main__":
    main()

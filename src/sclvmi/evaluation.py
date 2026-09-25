import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .context import sha256, timestamp, write_json
from .head import metrics


def evaluate(prediction_path, output_path, resamples=2000, seed=2025):
    output_path = Path(output_path)
    assert not output_path.exists(), "Use a new evaluation output path"
    table = pd.read_csv(prediction_path, dtype={"PatientID": str})
    assert table.identifier.is_unique
    assert table[["PatientID", "label", "probability"]].notna().all().all()
    assert table.probability.between(0, 1).all()
    assert set(table.label) == {0, 1}
    table = table.reset_index(drop=True)
    groups = list(table.groupby("PatientID", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    labels, probability = table.label.to_numpy(), table.probability.to_numpy()
    draws = []
    rejected = 0
    for _ in tqdm(range(resamples), desc="Patient bootstrap", mininterval=2):
        indices = np.concatenate([groups[index] for index in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(labels[indices])) < 2:
            rejected += 1
            continue
        draws.append(roc_auc_score(labels[indices], probability[indices]))
    assert len(draws) >= 0.9 * resamples, "Insufficient two-class patient bootstrap draws"
    write_json(output_path, {"completed_at": timestamp(), "prediction_sha256": sha256(prediction_path), "cases": len(table), "patients": len(groups), "metrics": metrics(labels, probability), "auroc_patient_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(), "resamples": resamples, "valid_resamples": len(draws), "rejected_single_class_draws": rejected, "seed": seed, "sampling_unit": "patient, retaining all associated lesions/timepoints"})
    np.save(output_path.with_suffix(".bootstrap.npy"), np.asarray(draws))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    evaluate(args.predictions, args.output, args.resamples, args.seed)

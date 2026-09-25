from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .context import context, sha256, timestamp, write_json


def feature_table(directory, split):
    table = pd.read_csv(directory / f"{split}_members.csv", dtype={"PatientID": str})
    features = []
    for identifier in table.identifier:
        with np.load(directory / f"{identifier}.npz") as data:
            features.append(data["embedding"])
    return table, np.stack(features)


def metrics(labels, probability):
    return {"auroc": roc_auc_score(labels, probability), "average_precision": average_precision_score(labels, probability), "brier": brier_score_loss(labels, probability), "log_loss": log_loss(labels, probability)}


def fit_head(feature_directory, run_id):
    storage, config, _ = context()
    directory = Path(feature_directory)
    output = Path(storage["runs"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    train, train_x = feature_table(directory, "train")
    dev, dev_x = feature_table(directory, "development")
    assert set(train.PatientID).isdisjoint(dev.PatientID)
    train.to_csv(output / "train_members.csv", index=False)
    dev.to_csv(output / "development_members.csv", index=False)
    results = []
    candidates = []
    for regularization in config["head"]["C_grid"]:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=regularization, class_weight=config["head"]["class_weight"], max_iter=config["head"]["max_iter"], random_state=config["seed"]))
        model.fit(train_x, train.label)
        probability = model.predict_proba(dev_x)[:, 1]
        score = metrics(dev.label, probability)
        results.append({"C": regularization, **score})
        candidates.append(model)
    selected = max(range(len(results)), key=lambda index: (results[index]["auroc"], -results[index]["log_loss"]))
    model = candidates[selected]
    joblib.dump(model, output / "head.joblib")
    reloaded = joblib.load(output / "head.joblib")
    predictions = reloaded.predict_proba(dev_x)[:, 1]
    np.testing.assert_array_equal(predictions, model.predict_proba(dev_x)[:, 1])
    dev.assign(probability=predictions).to_csv(output / "development_predictions.csv", index=False)
    write_json(output / "result.json", {"completed_at": timestamp(), "status": "completed", "feature_directory": str(directory.resolve()), "feature_config_sha256": sha256(directory / "config.json"), "configuration": config["head"], "selection": "development AUROC then log loss", "grid": results, "selected": results[selected], "train_cases": len(train), "development_cases": len(dev), "test_evaluated": False, "interpretation": "pipeline validation only when selected feature set is a smoke sample"})
    print(output, flush=True)
    return output

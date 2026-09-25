import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .context import context, sha256, timestamp, write_json
from .head import metrics


CONCEPTS = ["subtlety", "internal_structure", "calcification", "sphericity", "margin", "lobulation", "spiculation", "texture"]


def read_features(table, directory):
    vectors = []
    for identifier in table.identifier:
        with np.load(Path(directory) / f"{identifier}.npz") as values:
            vectors.append(values["embedding"])
    return np.stack(vectors)


def run(lidc_csv, lidc_features, luna_csv, luna_features, run_id):
    storage, config, _ = context()
    output = Path(storage["runs"]) / run_id
    output.mkdir(parents=True, exist_ok=False)
    lidc = pd.read_csv(lidc_csv, dtype={"PatientID": str})
    luna = pd.read_csv(luna_csv, dtype={"PatientID": str})
    required = ["identifier", "PatientID", *CONCEPTS]
    assert set(required).issubset(lidc.columns), required
    assert set(["identifier", "PatientID", "split", "label", "size_mm"]).issubset(luna.columns)
    assert lidc.identifier.is_unique and luna.identifier.is_unique
    assert lidc[CONCEPTS].notna().all().all()
    assert luna.size_mm.notna().all() and (luna.size_mm > 0).all()
    assert set(luna.split).issubset({"train", "development", "test"})
    assert luna.groupby("PatientID").split.nunique().max() == 1
    # Concept definitions and reader aggregation must be supplied by the data manifest.
    x = read_features(lidc, lidc_features)
    selected_luna = luna[luna.split.isin(["train", "development"])].copy()
    luna_x = read_features(selected_luna, luna_features)
    heads = []
    concept_report = []
    for concept in CONCEPTS:
        search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": np.logspace(-3, 4, 8)}, cv=GroupKFold(5), scoring="neg_mean_squared_error", n_jobs=1)
        search.fit(x, lidc[concept], groups=lidc.PatientID)
        heads.append(search.best_estimator_)
        concept_report.append({"concept": concept, "alpha": search.best_params_["ridge__alpha"], "selection_cv_mse": -search.best_score_, "note": "selection score; not an independent concept-fidelity estimate"})
    concept_values = np.column_stack([head.predict(luna_x) for head in heads] + [selected_luna.size_mm.to_numpy()])
    train = selected_luna.split.eq("train").to_numpy()
    dev = selected_luna.split.eq("development").to_numpy()
    classifier = ExplainableBoostingClassifier(interactions=0, random_state=config["seed"], n_jobs=1, validation_size=0)
    classifier.fit(concept_values[train], selected_luna.loc[train, "label"])
    probability = classifier.predict_proba(concept_values[dev])[:, 1]
    joblib.dump({"concept_heads": heads, "classifier": classifier, "concept_names": CONCEPTS, "size_definition": "external supplied size_mm; see data provenance"}, output / "cbm.joblib")
    selected_luna.loc[dev].assign(probability=probability).to_csv(output / "development_predictions.csv", index=False)
    write_json(output / "result.json", {"completed_at": timestamp(), "implementation_status": "independent reconstruction; author code and exact cohort mapping unavailable", "paper": "arXiv:2608.07857v1", "deviations": ["explicit standardized ridge grid", "patient-grouped CV", "InterpretML additive EBM with interactions=0 and validation_size=0; paper implementation unspecified"], "sources": {str(lidc_csv): sha256(lidc_csv), str(luna_csv): sha256(luna_csv)}, "concept_heads": concept_report, "development": metrics(selected_luna.loc[dev, "label"], probability), "test_evaluated": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for argument in ["lidc-csv", "lidc-features", "luna-csv", "luna-features", "run-id"]:
        parser.add_argument(f"--{argument}", required=True)
    args = parser.parse_args()
    run(args.lidc_csv, args.lidc_features, args.luna_csv, args.luna_features, args.run_id)

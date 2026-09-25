from pathlib import Path

import numpy as np
import pandas as pd

from .context import context, read_json, sha256, timestamp, write_json


def build_manifest():
    storage, config, dataset = context()
    clinical = pd.read_csv(dataset / "clinical_data.csv", dtype={"AnnotationID": str, "PatientID": str})
    labels = pd.read_csv(dataset / "cls_data.csv", dtype={"identifier": str})
    assert labels.identifier.is_unique and clinical.AnnotationID.is_unique
    merged = labels.merge(clinical, left_on="identifier", right_on="AnnotationID", validate="one_to_one", how="left", suffixes=("", "_clinical"))
    assert merged.PatientID.notna().all()
    assert np.array_equal(merged.label, merged.label_clinical)
    folds = read_json(dataset / "splits_final.json")
    mapping = {}
    for fold, members in enumerate(folds):
        for case in members["val"]:
            assert case not in mapping, case
            mapping[case] = fold
    assert set(mapping) == set(merged.identifier)
    merged["fold"] = merged.identifier.map(mapping)
    assert merged.groupby("PatientID").fold.nunique().max() == 1
    merged["split"] = merged.fold.map(lambda fold: "test" if fold == config["test_fold"] else "development" if fold == config["development_fold"] else "train")
    merged["image"] = merged.identifier.map(lambda case: str(dataset / "imagesTr" / f"{case}_0000.nii.gz"))
    merged["mask"] = merged.identifier.map(lambda case: str(dataset / "labelsTr" / f"{case}.nii.gz"))
    for column in ["image", "mask"]:
        assert all(Path(path).is_file() for path in merged[column])
    columns = ["identifier", "PatientID", "SeriesInstanceUID", "NoduleID", "StudyDate", "label", "fold", "split", "image", "mask"]
    output = Path(storage["preprocessed"]) / "luna25" / config["dataset_revision"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.csv"
    table = merged[columns].sort_values("identifier")
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path, dtype={"PatientID": str}), table.reset_index(drop=True), check_dtype=False)
    else:
        table.to_csv(path, index=False)
    receipt = {"created_at": timestamp(), "revision": config["dataset_revision"], "source_hashes": {name: sha256(dataset / name) for name in ["clinical_data.csv", "cls_data.csv", "splits_final.json", "dataset.json"]}, "manifest_sha256": sha256(path), "counts": table.groupby("split").agg(cases=("identifier", "size"), patients=("PatientID", "nunique"), malignant=("label", "sum")).to_dict("index"), "clinical_rows_without_images": len(clinical) - len(table)}
    write_json(output / "manifest.json", receipt)
    print(receipt["counts"], flush=True)
    return path


def load_manifest():
    storage, config, _ = context()
    path = Path(storage["preprocessed"]) / "luna25" / config["dataset_revision"] / "manifest.csv"
    return pd.read_csv(path, dtype={"PatientID": str})


def select_cases(table, split, limit=None):
    selected = table[table.split == split].sort_values("identifier")
    if limit:
        assert limit >= 2
        parts = [selected[selected.label == label].head(limit // 2) for label in [0, 1]]
        selected = pd.concat(parts).sort_values("identifier")
    assert len(selected) and selected.label.nunique() == 2
    return selected

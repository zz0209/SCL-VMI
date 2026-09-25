import argparse
import gc
import importlib.metadata
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.data import load_manifest
from sclvmi.evaluation import evaluate
from sclvmi.predict import fm_predict, i3d_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--i3d-run", required=True)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    storage, configuration, _ = context()
    runs = Path(storage["runs"])
    output = runs / args.output_id
    output.mkdir(parents=True, exist_ok=False)
    table = load_manifest()
    results = {}
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for name in configuration["models"]:
            directory = runs / (args.baseline_run + "_" + name)
            result = read_json(directory / "result.json")
            features = Path(result["feature_directory"])
            train = pd.read_csv(directory / "train_members.csv", dtype={"PatientID": str})
            dev = pd.read_csv(directory / "development_predictions.csv", dtype={"PatientID": str})
            for split, members in [("train", train), ("development", dev)]:
                expected = table[table.split == split]
                assert set(members.identifier) == set(expected.identifier)
                assert set(members.PatientID) == set(expected.PatientID)
            assert set(train.PatientID).isdisjoint(dev.PatientID)
            assert not (features / "test_members.csv").exists()
            assert len(list(features.glob("*.npz"))) == len(train) + len(dev)
            head = joblib.load(directory / "head.joblib")
            iterations = int(head[-1].n_iter_.max())
            assert iterations < head[-1].max_iter, "Selected head did not converge"
            row = dev.iloc[0]
            prediction = fm_predict(name, directory, row.image)
            np.testing.assert_allclose(prediction["malignancy_probability"], row.probability, atol=1e-5, rtol=1e-5)
            evaluate(directory / "development_predictions.csv", output / f"{name}_development.json")
            results[name] = {"single_image_matches_cached_prediction": True, "solver_iterations": iterations, "train_cases": len(train), "development_cases": len(dev), "test_used": False, "head_sha256": sha256(directory / "head.joblib"), "feature_config_sha256": sha256(features / "config.json")}
            print(f"Validated full pipeline: {name}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        i3d = runs / args.i3d_run
        history = read_json(i3d / "history.json")
        selected = max(history, key=lambda item: item["auroc"])
        predictions = pd.read_csv(i3d / f"development_epoch_{selected['epoch']:03d}.csv")
        row = predictions.iloc[0]
        prediction = i3d_predict(i3d / "best.pt", row.image)
        np.testing.assert_allclose(prediction["malignancy_probability"], row.probability, atol=1e-5, rtol=1e-5)
        results["i3d"] = {"single_image_matches_training_validation": True, "checkpoint_sha256": sha256(i3d / "best.pt")}
    write_json(output / "verification.json", {"completed_at": timestamp(), "status": "completed", "models": results, "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in sorted((ROOT / "src/sclvmi").glob("*.py"))}, "configuration": configuration, "python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["torch", "monai", "numpy", "scipy", "scikit-learn", "SimpleITK", "nibabel"]}, "gpu": torch.cuda.get_device_name(), "baseline_run": args.baseline_run, "i3d_run": args.i3d_run, "interpretation": "Development metrics include head selection; bootstrap intervals do not remove selection effects"})


if __name__ == "__main__":
    main()

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sclvmi.context import context, read_json, timestamp, write_json
from sclvmi.sae_campaign import campaign_root, configuration
from sclvmi.sae_predict import SpatialSAEPipeline
from sclvmi.sae_sites import SITES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    root = campaign_root()
    storage, _, _ = context()
    records = []
    for run_id in args.runs:
        run = root / "runs" / run_id
        pipeline = SpatialSAEPipeline(run / "dictionary.pt")
        model, site = pipeline.name, pipeline.site
        old = Path(storage["activations"]) / configuration()["previous_suite"] / model
        members = pd.read_csv(old / "full_members.csv", dtype={"PatientID": str}).query("split == 'development'")
        chosen = members.sort_values("identifier").groupby("label", sort=True).head(1)
        assert len(chosen) == 2
        for row in chosen.itertuples():
            bundle = pipeline.extract(row.image)
            source = old / "cases" if site in SITES[model] else root / "cases" / model
            with np.load(source / f"{row.identifier}.npz") as saved:
                raw = torch.from_numpy(saved[site]).cuda()
                probability = float(saved["probability"][0])
            with torch.inference_mode():
                expected = pipeline.dictionary.spatial_activations(raw).cpu().numpy()
            np.testing.assert_allclose(bundle["activations"], expected, rtol=3e-5, atol=3e-5)
            assert abs(bundle["geometry"]["original_probability"] - probability) < 1e-5
            if site == "global":
                assert bundle["activations"].shape[1:] == (1, 1, 1)
                assert bundle["geometry"]["grid_affine_ras"] is None
            else:
                expected_affine = np.asarray(bundle["geometry"]["input_affine_ras"]) @ np.diag([pipeline.specification["stride"]] * 3 + [1])
                np.testing.assert_array_equal(bundle["geometry"]["grid_affine_ras"], expected_affine)
            downstream_path = run / "downstream_full/predictions.csv"
            if downstream_path.exists():
                frame = pd.read_csv(downstream_path)
                expected_probability = frame.loc[frame.identifier == row.identifier, "recovered"]
                assert len(expected_probability) == 1
                assert abs(bundle["geometry"]["reconstructed_probability"] - expected_probability.item()) < 1e-5
            records.append({"run_id": run_id, "case": row.identifier, "maximum_activation_error": float(np.abs(bundle["activations"] - expected).max()), "geometry": bundle["geometry"]})
        print("RAW_PREDICTION_CHECK", run_id, len(chosen), flush=True)
        del pipeline, raw, bundle
        gc.collect()
        torch.cuda.empty_cache()
    receipt = {"state": "passed", "records": records, "test_used": False, "created_at": timestamp()}
    write_json(root / "smoke" / f"raw_prediction_{time.time_ns()}.json", receipt)
    print("RAW_PREDICTION_PASSED", len(records), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    main()

import argparse
from pathlib import Path

import numpy as np
import torch

from sclvmi.context import read_json, timestamp, write_json
from sclvmi.sae_campaign import campaign_root, configuration
from sclvmi.sae_campaign_train import fit_job


def job(name, site, tag):
    return {
        "run_id": f"smoke_{name}_{site}_{tag}_s2025", "model": name, "site": site,
        "seed": 2025, "k": 64, "expansion": 4, "matrix_directory": "smoke_matrices",
        "training": {"steps": 48, "batch_size": 64, "validate_every": 24, "checkpoint_every": 24},
        "inference": {"calibration_train_tokens": 256},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    root = args.root or campaign_root() / "smoke"
    records = []
    for name, sites in configuration()["models"].items():
        for site in sites:
            if args.only and f"{name}/{site}" not in args.only:
                continue
            specification = job(name, site, "complete")
            result = fit_job(root, specification, "smoke")
            assert result["state"] == "completed"
            assert all(np.isfinite(result["development"][key]) for key in ["fvu", "cosine", "l0_mean"])
            records.append({"model": name, "site": site, "peak_gpu_bytes": result["peak_gpu_bytes"], "elapsed_seconds": result["elapsed_seconds"]})
    if not args.only or "vista/stage2" in args.only:
        resumed = job("vista", "stage2", "resume")
        resume_path = root / "runs" / resumed["run_id"]
        if not (resume_path / "result.json").exists():
            paused = fit_job(root, resumed, "smoke", stop_after=23)
            assert paused["state"] == "paused" and paused["step"] == 23
            checkpoint = torch.load(resume_path / "resume.pt", map_location="cpu", weights_only=False)
            assert checkpoint["step"] == 23 and checkpoint["optimizer"]["state"]
            fit_job(root, resumed, "smoke")
        reference_path = root / "runs" / job("vista", "stage2", "complete")["run_id"]
        reference = torch.load(reference_path / "resume.pt", map_location="cpu", weights_only=False)
        recovered = torch.load(resume_path / "resume.pt", map_location="cpu", weights_only=False)
        for key, value in reference["state_dict"].items():
            torch.testing.assert_close(value, recovered["state_dict"][key], rtol=0, atol=0)
        torch.testing.assert_close(reference["generator"], recovered["generator"], rtol=0, atol=0)
        torch.testing.assert_close(reference["inactive"], recovered["inactive"], rtol=0, atol=0)
        original_result = read_json(reference_path / "result.json")
        resumed_result = read_json(resume_path / "result.json")
        assert original_result["development"] == resumed_result["development"]
        records.append({"resume": "exact parameters, generator, activity counters and development metrics"})
    write_json(root / "training_witness.json", {"state": "passed", "records": records, "created_at": timestamp(), "input": "actual training and development patient vectors"})
    print("CAMPAIGN_TRAINING_WITNESS", records, flush=True)


if __name__ == "__main__":
    main()

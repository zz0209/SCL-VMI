import argparse
import shutil
from pathlib import Path

import pandas as pd

from sclvmi.context import ROOT, read_json, sha256, timestamp, write_json


def public(value):
    # machine paths (the deployed head run, matrix and storage locations) stay private; run names and numbers are kept
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items() if key not in {"head_run"}}
    if isinstance(value, list):
        return [public(item) for item in value]
    if isinstance(value, str):
        assert not value.startswith("/") and ":\\" not in value, value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True, help="the campaign report directory copied from the workstation")
    parser.add_argument("--output", type=Path, default=ROOT / "results/20261021_flexict_sae")
    args = parser.parse_args()
    config = read_json(ROOT / "configs/flexict_sae.json")
    summary = read_json(args.report / "summary.json")
    assert summary["campaign"] == config["campaign"] and not summary["test_used"], summary["campaign"]
    args.output.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(args.report / "runs.csv")
    groups = pd.read_csv(args.report / "groups.csv")
    assert not runs.astype(str).apply(lambda column: column.str.startswith("/")).any().any() and not groups.astype(str).apply(lambda column: column.str.startswith("/")).any().any()
    runs.to_csv(args.output / "runs.csv", index=False)
    groups.to_csv(args.output / "groups.csv", index=False)
    write_json(args.output / "summary.json", public({"campaign": summary["campaign"], "verdicts": summary["verdicts"], "quality_rule": summary["quality_rule"], "selection": summary["selection"],
                                                     "acceptance_revision": config["acceptance_revision"], "test_used": False}))
    for name in ["classifier_gate.png", "classifier_gate.svg", "sae_quality.png", "sae_quality.svg"]:
        shutil.copyfile(args.report / name, args.output / name)
    write_json(args.output / "provenance.json", {"campaign": config["campaign"], "features_suite": config["features_suite"], "report_created_at": summary["created_at"],
                                                 "report_code_sha256": summary["code_sha256"], "report_config_sha256": summary["config_sha256"],
                                                 "export_config_sha256": sha256(ROOT / "configs/flexict_sae.json"), "exported_at": timestamp(), "test_used": False})
    print("EXPORTED", args.output, len(runs), "runs", len(groups), "groups", flush=True)


if __name__ == "__main__":
    main()

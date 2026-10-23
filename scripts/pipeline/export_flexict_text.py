import argparse
import shutil
from pathlib import Path

import pandas as pd

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


def public(value):
    # run names and numbers are kept; machine paths stay private
    if isinstance(value, dict):
        return {key: public(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public(item) for item in value]
    if isinstance(value, str):
        assert not value.startswith("/") and ":\\" not in value, value
    return value


def table(source, destination):
    frame = pd.read_csv(source)
    assert not frame.astype(str).apply(lambda column: column.str.contains(r":\\|^/", regex=True)).any().any(), source
    assert not {"identifier", "PatientID", "SeriesInstanceUID", "record"} & set(frame.columns), source
    frame.to_csv(destination, index=False)
    return len(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/20261023_flexict_text")
    args = parser.parse_args()
    config = read_json(ROOT / "configs/flexict_text.json")
    run = Path(context()[0]["runs"]) / config["run_id"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = {}
    for source, name in [("calibration/tests.csv", "text_tests.csv"), ("calibration/concepts.csv", "usable_concepts.csv"), ("calibration_specific/tests.csv", "text_tests_specific.csv"),
                         ("calibration_specific/concepts.csv", "usable_concepts_specific.csv"), ("naming/features.csv", "w20_naming.csv"), ("bridge/f32_b10.csv", "bridge_f32_b10.csv"),
                         ("bridge/f40_b08.csv", "bridge_f40_b08.csv")] + [(f"contributions/{site}_features.csv", f"contributions_{site}.csv") for site in config["contributions"]["sites"]]:
        rows[name] = table(run / source, args.output / name)
    for directory, suffix in [("calibration", ""), ("calibration_specific", "_specific")]:
        for level in ["scan", "token"]:
            for extension in ["png", "svg"]:
                shutil.copyfile(run / directory / f"{level}_tests.{extension}", args.output / f"{level}_tests{suffix}.{extension}")
    calibration, specific = read_json(run / "calibration/summary.json"), read_json(run / "calibration_specific/summary.json")
    naming, bridge, contributions = read_json(run / "naming/summary.json"), read_json(run / "bridge/summary.json"), read_json(run / "contributions/summary.json")
    for item in [calibration, specific, naming, bridge, contributions]:
        assert not item["test_used"]
    write_json(args.output / "summary.json", public({
        "run": config["run_id"], "patients": calibration["patients"], "exclusions": calibration["exclusions"],
        "step1_usable": calibration["usable"], "step1_tests": calibration["tests"], "step1_specific_usable": specific["usable"], "step1_specific_tests": specific["tests"],
        "step2_naming": {key: naming[key] for key in ["stable_features", "token_concepts", "scan_concepts", "named_features", "names", "reference_passing_before_cross_seed"]},
        "step5_bridge": {site: bridge[site] for site in config["bridge"]["sites"]},
        "step4_contributions": {site: {key: contributions[site][key] for key in ["group", "head_run", "records", "probability_max_abs_difference", "auroc", "groups"] + (["concept_model"] if "concept_model" in contributions[site] else [])}
                                for site in config["contributions"]["sites"]},
        "rules": {key: config[key] for key in ["pass_rule", "usability_rule", "specific_variant", "naming", "bridge", "contributions"]}, "test_used": False}))
    write_json(args.output / "provenance.json", {"run": config["run_id"], "sae_campaign": config["sae_campaign"], "inputs_suite": config["inputs_suite"], "rows": rows,
                                                 "sources": {name: {"created_at": item["created_at"], "code_sha256": item["code_sha256"], "config_sha256": item["config_sha256"]}
                                                             for name, item in [("calibration", calibration), ("calibration_specific", specific), ("naming", naming), ("bridge", bridge), ("contributions", contributions)]},
                                                 "export_config_sha256": sha256(ROOT / "configs/flexict_text.json"), "exported_at": timestamp(), "test_used": False})
    print("EXPORTED", args.output, rows, flush=True)


if __name__ == "__main__":
    main()

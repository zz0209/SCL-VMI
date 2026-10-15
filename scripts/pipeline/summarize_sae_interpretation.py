import argparse
import json
from pathlib import Path

import numpy as np

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


def quantile(values, level):
    clean = values[np.isfinite(values)]
    return float(np.quantile(clean, level))


def site_summary(root, config, choice):
    reference = config["grid"]["reference_seed"]
    trained = read_json(root / "stability" / choice["group"] / "summary.json")
    control = read_json(root / "stability" / choice["control_group"] / "summary.json")
    with np.load(root / "stability" / choice["group"] / "features.npz") as saved:
        stable = (saved["reappearance"][0] >= config["stability"]["stable_minimum_reappearance"]) & saved["development_live"][0]
    with np.load(root / "profiles" / f"{choice['model']}_{choice['site']}" / "profiles.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    prefix, random_prefix = f"trained_{reference}_", f"random_{reference}_"
    enrichment_limit = quantile(arrays[random_prefix + "nodule_enrichment"], 0.95)
    partial_limit = quantile(np.abs(arrays[random_prefix + "partial_development"]), 0.95)
    enrichment = arrays[prefix + "nodule_enrichment"]
    discovery, confirmation = arrays[prefix + "partial_train"], arrays[prefix + "partial_development"]
    enriched = stable & (enrichment > enrichment_limit)
    associated = stable & (np.abs(discovery) > partial_limit) & (np.abs(confirmation) > partial_limit) & (np.sign(discovery) == np.sign(confirmation))
    runs = [item["run_id"] for item in trained["seeds"]]
    downstream = [read_json(root / "runs" / run / "downstream.json") for run in runs]
    return {
        "site": f"{choice['model']}:{choice['site']}", "expansion": choice["expansion"], "k": choice["k"], "training_steps": choice["training_steps"],
        "features": int(len(stable)),
        "trained_stable_per_seed": [item["stable_features"] for item in trained["seeds"]], "random_stable_per_seed": [item["stable_features"] for item in control["seeds"]],
        "trained_stable_explained_variance": trained["reference_stable_explained_variance"], "random_stable_explained_variance": control["reference_stable_explained_variance"],
        "trained_fvu_range": [min(item["sample_fvu"] for item in trained["seeds"]), max(item["sample_fvu"] for item in trained["seeds"])], "pca_fvu": trained["pca_reference"]["sample_fvu"],
        "trained_pw_mcc": trained["pairs"][0]["pw_mcc_decoder_cosine"], "random_pw_mcc": control["pairs"][0]["pw_mcc_decoder_cosine"],
        "probability_mae_range": [min(item["metrics"]["probability_mae"] for item in downstream), max(item["metrics"]["probability_mae"] for item in downstream)],
        "recovered_auroc_range": [min(item["metrics"]["recovered_auroc"] for item in downstream), max(item["metrics"]["recovered_auroc"] for item in downstream)],
        "original_auroc": downstream[0]["metrics"]["original_auroc"],
        "interior_probability_mae_range": [min(item["interior_replacement"]["metrics"]["probability_mae"] for item in downstream), max(item["interior_replacement"]["metrics"]["probability_mae"] for item in downstream)] if "interior_replacement" in downstream[0] else None,
        "reference_seed_stable_features": int(stable.sum()),
        "random_95th_nodule_enrichment": enrichment_limit, "stable_above_random_95th_enrichment": int(enriched.sum()),
        "random_95th_abs_partial_confirmation": partial_limit, "stable_with_concordant_partial_above_random_95th": int(associated.sum()),
        "max_single_feature_development_auroc_trained": float(np.nanmax(arrays[prefix + "development_auroc"])), "max_single_feature_development_auroc_random": float(np.nanmax(arrays[random_prefix + "development_auroc"])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/sae_interpretation.json")
    root = Path(context()[0]["runs"]) / config["campaign"]
    selection = read_json(root / "selection.json")
    rows = [site_summary(root, config, choice) for choice in selection["choices"].values()]
    size = read_json(root / "profiles" / "fmcib_layer2" / "summary.json")["size_only_development_auroc"]
    value = {"sites": rows, "size_only_development_auroc": size, "rules": {"stable": "reappears with decoder cosine >= 0.7 in at least 6 of 7 other seeds and is live on development", "enrichment": "nodule-region enrichment above the 95th percentile of the random-weight reference-seed features", "association": "absolute size-adjusted Spearman partial correlation above the random 95th percentile in both discovery and confirmation patients, same sign"}, "selection_sha256": sha256(root / "selection.json"), "created_at": timestamp(), "test_used": False}
    destination = args.output or root / "summary" / "production_summary.json"
    write_json(destination, value)
    print(json.dumps(value, indent=1))


if __name__ == "__main__":
    main()

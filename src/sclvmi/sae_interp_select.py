import argparse
from pathlib import Path

from .context import ROOT, context, read_json, sha256, timestamp, write_json


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
SITES = [("fmcib", "layer1_context"), ("fmcib", "layer2"), ("vista", "stage2"), ("vista", "stage3")]
SOURCES = {("fmcib", "layer1_context"): "fmcib_context", ("fmcib", "layer2"): "fmcib", ("vista", "stage2"): "vista", ("vista", "stage3"): "vista"}


def run_id(config, encoder, model, site, expansion, k, seed, variant=None):
    value = f"{config['campaign']}_{encoder}_{model}_{site}_e{expansion}_k{k}_s{seed}"
    return f"{value}_{variant}" if variant else value


def configuration_row(root, config, encoder, model, site, expansion, k, variant=None, directory="stability"):
    group = f"{encoder}_{model}_{site}_e{expansion}_k{k}" + (f"_{variant}" if variant else "")
    reference = config["grid"]["reference_seed"]
    summary_path = root / directory / group / "summary.json"
    run = root / "runs" / run_id(config, encoder, model, site, expansion, k, reference, variant)
    row = {"group": group, "encoder": encoder, "model": model, "site": site, "expansion": expansion, "k": k, "variant": variant or "tokens1x", "features": None, "stability_ready": summary_path.exists()}
    if (run / "result.json").exists():
        result = read_json(run / "result.json")
        row.update({"features": result["request"]["dict_size"], "training_steps": result["request"]["training"]["steps"], "development_fvu": result["development"]["fvu"], "development_cosine": result["development"]["cosine"], "inactive_fraction": result["development"]["inactive_fraction"], "l0_mean": result["development"]["l0_mean"]})
    if summary_path.exists():
        summary = read_json(summary_path)
        first = summary["seeds"][0]
        row.update({
            "seeds": len(summary["seeds"]), "live_features": first["live_features"], "stable_features": first["stable_features"], "unstable_features": first["unstable_features"],
            "stable_fraction": first["stable_fraction_of_live"], "stable_explained_variance": summary["reference_stable_explained_variance"],
            "sample_fvu": first["sample_fvu"], "pca_fvu": summary["pca_reference"]["sample_fvu"], "pw_mcc": summary["pairs"][0]["pw_mcc_decoder_cosine"],
            "mean_stable_fraction_all_seeds": sum(item["stable_fraction_of_live"] for item in summary["seeds"]) / len(summary["seeds"]),
            "unstable_subspace_energy": summary["unstable_subspace"].get("mean_captured_energy"),
            "unstable_subspace_chance": (sum(item["chance_energy"] for item in summary["unstable_subspace"]["pairs"]) / len(summary["unstable_subspace"]["pairs"])) if summary["unstable_subspace"]["pairs"] else None,
        })
    if encoder == "trained" and (run / "downstream.json").exists():
        downstream = read_json(run / "downstream.json")
        row.update({"probability_mae": downstream["metrics"]["probability_mae"], "original_auroc": downstream["metrics"]["original_auroc"], "recovered_auroc": downstream["metrics"]["recovered_auroc"], "original_ap": downstream["metrics"]["original_ap"], "recovered_ap": downstream["metrics"]["recovered_ap"], "classifier_pass": downstream["checks_pass"]})
        if "interior_replacement" in downstream:
            row["interior_probability_mae"] = downstream["interior_replacement"]["metrics"]["probability_mae"]
    return row


def grid_table(root, config):
    grid = config["grid"]
    return [configuration_row(root, config, encoder, model, site, expansion, k) for model, site in SITES for encoder in ["trained", "random"] for k in grid["k"] for expansion in grid["expansion"]]


def budget_table(root, config):
    rows = []
    for variant in config["budget_diagnostic"]["variants"]:
        for model, site, expansion, k in variant["configurations"]:
            for name in [None, variant["name"]]:
                row = configuration_row(root, config, "trained", model, site, expansion, k, name, "stability_budget")
                if row["stability_ready"] and all(item["group"] != row["group"] for item in rows):
                    rows.append(row)
    return rows


def select(root, config):
    production = config["production"]
    grid = grid_table(root, config)
    choices = {}
    for model, site in SITES:
        trained = [row for row in grid if row["encoder"] == "trained" and row["model"] == model and row["site"] == site]
        assert all(row["stability_ready"] and "classifier_pass" in row for row in trained), f"{model}/{site} lacks stability or classifier evidence"
        eligible = [row for row in trained if row["classifier_pass"]]
        assert eligible, f"No {model}/{site} configuration passes the fixed-classifier checks"
        best = max(eligible, key=lambda row: (row["stable_explained_variance"], row["stable_features"]))
        site_choice = production["sites"][f"{model}:{site}"]
        variant, expansion, k = site_choice["variant"], site_choice["expansion"], site_choice["k"]
        assert k == min(row["k"] for row in eligible if row["expansion"] == expansion), f"{model}/{site} production k is not the smallest passing value"
        chosen = configuration_row(root, config, "trained", model, site, expansion, k, variant)
        control = configuration_row(root, config, "random", model, site, expansion, k, variant)
        assert chosen["stability_ready"] and control["stability_ready"], f"{model}/{site} production stability is incomplete"
        assert chosen.get("classifier_pass"), f"{model}/{site} production reference seed fails or lacks the fixed-classifier checks"
        choices[f"{model}:{site}"] = {
            "model": model, "site": site, "expansion": expansion, "k": k, "variant": variant, "training_steps": chosen["training_steps"],
            "group": chosen["group"], "control_group": control["group"], "grid_best_group": best["group"],
            "stable_explained_variance": chosen["stable_explained_variance"], "stable_features": chosen["stable_features"], "live_features": chosen["live_features"], "probability_mae": chosen["probability_mae"],
            "control_stable_explained_variance": control["stable_explained_variance"], "control_stable_features": control["stable_features"], "control_live_features": control["live_features"],
            "trained": chosen, "control": control,
            "basis": production["basis"],
        }
    return grid, budget_table(root, config), choices


def publish(root, config, choices):
    entries = []
    for choice in choices.values():
        model, site = choice["model"], choice["site"]
        for encoder, group_name, group in [("trained", "interpretation", choice["group"]), ("random", "control", choice["control_group"])]:
            summary = read_json(root / "stability" / group / "summary.json")
            for seed_summary in summary["seeds"]:
                run = seed_summary["run_id"]
                result = read_json(root / "runs" / run / "result.json")
                assert sha256(root / "runs" / run / "dictionary.pt") == result["dictionary_sha256"] == summary["dictionaries"][run]
                entry = {
                    "run_id": run, "model": model, "site": site, "seed": seed_summary["seed"], "features": result["request"]["dict_size"], "k": choice["k"], "expansion": choice["expansion"],
                    "variant": choice["variant"], "training_steps": result["request"]["training"]["steps"], "stability_group": group,
                    "encoder": encoder, "group": group_name, "family": f"{group_name}:{model}:{site}", "image_source": SOURCES[(model, site)], "dictionary_sha256": result["dictionary_sha256"],
                    "development": result["development"],
                    "quality": {"fvu": result["development"]["fvu"], "pca_fvu": summary["pca_reference"]["sample_fvu"], "stable_features": seed_summary["stable_features"], "live_features": seed_summary["live_features"], "stable_explained_variance": 1 - seed_summary["stable_feature_fvu"], "pw_mcc": summary["pairs"][0]["pw_mcc_decoder_cosine"]},
                }
                if encoder == "trained":
                    downstream = read_json(root / "runs" / run / "downstream.json")
                    assert downstream["checks_pass"], run
                    entry["downstream"] = downstream["metrics"]
                    if "interior_replacement" in downstream:
                        entry["interior_downstream"] = downstream["interior_replacement"]["metrics"]
                entries.append(entry)
    entries.sort(key=lambda row: (["interpretation", "control"].index(row["group"]), [f"{m}:{s}" for m, s in SITES].index(f"{row['model']}:{row['site']}"), row["seed"] != config["grid"]["reference_seed"], row["seed"]))
    value = {"state": "completed", "dictionaries": entries, "created_at": timestamp(), "test_used": False}
    write_json(root / "viewer/index.json", value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    grid, budget, choices = select(root, config)
    write_json(root / "selection.json", {"state": "selected", "choices": choices, "configurations": grid, "budget_diagnostic": budget, "rule": config["selection"], "production": config["production"], "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    for key, choice in choices.items():
        print("SELECTED", key, choice["group"], choice["stable_features"], round(choice["stable_explained_variance"], 3), flush=True)
    if args.publish:
        print("PUBLISHED", len(publish(root, config, choices)["dictionaries"]), flush=True)


if __name__ == "__main__":
    main()

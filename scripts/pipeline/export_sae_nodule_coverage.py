import argparse
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


SITES = [("fmcib", "layer2", "FMCIB layer2"), ("vista", "stage2", "VISTA3D stage2"), ("vista", "stage3", "VISTA3D stage3")]
VARIANT_ORDER = ["tokens4x", "tokens16x", "nodule4x", "nodule16x", "nodule75"]
REGIONS = ["nodule_0_4mm", "near_4_10mm", "far_over_10mm"]
COLORS = {"nodule": "#2a78d6", "far": "#8d8b86", "selected": "#eb6834"}
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def coverage_row(root, path, variants):
    value = read_json(path)
    head = value["group"].split("_")
    encoder, model = head[0], head[1]
    variant = head[-1]
    expansion, k = int(head[-3][1:]), int(head[-2][1:])
    site = "_".join(head[2:-3])
    training = variants[variant]
    row = {"encoder": encoder, "model": model, "site": site, "expansion": expansion, "k": k, "variant": variant, "training_steps": training["steps"],
           "nodule_pool_fraction": training.get("nodule_pool", {}).get("fraction", 0.0), "seeds": len(value["seeds"]), "stable_rule": value["stable_rule"],
           "stable_features_reference_seed": value["stable_features"], "patients": value["patients"]}
    for region in REGIONS:
        row.update({f"{region}_tokens": value["regions"][region]["tokens"], f"{region}_explained_variance": value["regions"][region]["explained_variance"],
                    f"{region}_stable_explained_variance": value["regions"][region]["stable_explained_variance"], f"{region}_stable_activation_share": value["regions"][region]["stable_activation_share"]})
    row.update({"nodule_minus_far": value["nodule_minus_far"], "nodule_minus_far_low": value["interval"]["nodule_minus_far_low"], "nodule_minus_far_high": value["interval"]["nodule_minus_far_high"]})
    for threshold, counts in value["nodule_enriched"].items():
        label = threshold.replace(".0", "")
        row.update({f"enrichment_ge_{label}_features": counts["features"], f"enrichment_ge_{label}_stable": counts["stable"]})
    downstream = root / "runs" / value["run_id"] / "downstream.json"
    if encoder == "trained":
        metrics = read_json(downstream)
        row.update({"probability_mae_reference_seed": metrics["metrics"]["probability_mae"], "auroc_drop_reference_seed": metrics["metrics"]["original_auroc"] - metrics["metrics"]["recovered_auroc"],
                    "ap_drop_reference_seed": metrics["metrics"]["original_ap"] - metrics["metrics"]["recovered_ap"], "classifier_pass_reference_seed": metrics["checks_pass"]})
    return row


def order_key(row):
    standard = (row.expansion, row.k) == (4, 16)
    return (not standard, VARIANT_ORDER.index(row.variant), -row.expansion, row.k)


def figure(comparison, chosen, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    counts = [len(comparison[(comparison.model == model) & (comparison.site == site)]) for model, site, _ in SITES]
    fig, axes = plt.subplots(3, 3, figsize=(17, 9.6), facecolor=SURFACE, gridspec_kw={"width_ratios": counts, "height_ratios": [1.3, 1, 1]})
    for column, (model, site, title) in enumerate(SITES):
        rows = comparison[(comparison.model == model) & (comparison.site == site)]
        rows = rows.iloc[sorted(range(len(rows)), key=lambda index: order_key(rows.iloc[index]))]
        positions = np.arange(len(rows))
        labels = [f"{row.variant}\ne{row.expansion} k{row.k}" for row in rows.itertuples()]
        selected = [chosen.get(f"{model}:{site}") == {"variant": row.variant, "expansion": row.expansion, "k": row.k} for row in rows.itertuples()]
        coverage, enriched, mae = axes[0, column], axes[1, column], axes[2, column]
        coverage.vlines(positions, rows.nodule_0_4mm_stable_explained_variance, rows.far_over_10mm_stable_explained_variance, color=GRID, linewidth=3)
        coverage.plot(positions, rows.far_over_10mm_stable_explained_variance, "o", color=COLORS["far"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label="positions > 10 mm from the nodule surface")
        coverage.plot(positions, rows.nodule_0_4mm_stable_explained_variance, "o", color=COLORS["nodule"], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label="nodule and surface margin ≤ 4 mm")
        coverage.axhline(0, color=MUTED, linewidth=0.8)
        coverage.set_title(title, fontsize=10, loc="left")
        enriched.bar(positions, rows.enrichment_ge_3_stable, width=0.55, color=[COLORS["selected"] if flag else COLORS["nodule"] for flag in selected], edgecolor=SURFACE, linewidth=1)
        for position, count in zip(positions, rows.enrichment_ge_3_stable):
            enriched.text(position, count, f"{count}", ha="center", va="bottom", fontsize=8, color=INK)
        mae.bar(positions, rows.probability_mae_reference_seed, width=0.55, color=[COLORS["selected"] if flag else COLORS["nodule"] for flag in selected], edgecolor=SURFACE, linewidth=1)
        mae.axhline(0.025, color=MUTED, linewidth=1, linestyle="--")
        mae.text(positions[-1] + 0.3, 0.025, "gate 0.025", ha="right", va="bottom", fontsize=8, color=MUTED)
        mae.set_xticks(positions, labels)
        for axis in [coverage, enriched]:
            axis.set_xticks(positions, [])
        for position, flag in zip(positions, selected):
            if flag:
                coverage.annotate("selected", (position, max(rows.nodule_0_4mm_stable_explained_variance.max(), rows.far_over_10mm_stable_explained_variance.max())), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=INK)
        for axis in [coverage, enriched, mae]:
            axis.set_facecolor(SURFACE)
            axis.grid(axis="y", color=GRID, linewidth=0.6)
            axis.set_axisbelow(True)
            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
        if column == 0:
            coverage.set_ylabel("Variance explained by stable\nfeatures (development patients)")
            enriched.set_ylabel("Stable features with\nnodule enrichment ≥ 3")
            mae.set_ylabel("Fixed-classifier probability\nMAE (seed 2025)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.005, 0.955), frameon=False, ncol=2)
    fig.suptitle("Nodule-region training, seeds 2025-2028: stable = reappears in all 3 other seeds; selected configurations were retrained with 8 seeds", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"coverage.{suffix}", dpi=180, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/sae_interpretation.json")
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    destination = args.output
    destination.mkdir(parents=True, exist_ok=False)
    variants = {variant["name"]: variant["training"] for variant in config["budget_diagnostic"]["variants"]}
    sites = {f"{model}_{site}" for model, site, _ in SITES}
    comparison = pd.DataFrame([coverage_row(root, path, variants) for path in sorted((root / "coverage").glob("stability_budget_*.json"))])
    comparison = comparison[(comparison.model + "_" + comparison.site).isin(sites)].reset_index(drop=True)
    comparison.to_csv(destination / "comparison.csv", index=False)
    chosen = config["nodule_coverage"]["chosen"]
    production = pd.DataFrame([coverage_row(root, path, variants) for path in sorted((root / "coverage").glob("stability_*.json")) if not path.name.startswith("stability_budget_")])
    index = pd.DataFrame(read_json(root / "viewer/index.json")["dictionaries"])
    ranges = []
    for row in production.itertuples():
        entries = index[(index.encoder == row.encoder) & (index.model == row.model) & (index.site == row.site) & (index.variant == row.variant) & (index.k == row.k) & (index.expansion == row.expansion)]
        if len(entries):
            stable = entries.quality.map(lambda value: value["stable_features"])
            value = {"published": True, "stable_features_min": int(stable.min()), "stable_features_max": int(stable.max())}
            if row.encoder == "trained":
                metrics = entries.downstream
                value.update({"probability_mae_min": metrics.map(lambda item: item["probability_mae"]).min(), "probability_mae_max": metrics.map(lambda item: item["probability_mae"]).max(),
                              "recovered_auroc_min": metrics.map(lambda item: item["recovered_auroc"]).min(), "recovered_auroc_max": metrics.map(lambda item: item["recovered_auroc"]).max(), "seeds_passing_classifier": int(len(entries))})
        else:
            stable = [item["stable_features"] for item in read_json(root / "stability" / f"{row.encoder}_{row.model}_{row.site}_e{row.expansion}_k{row.k}_{row.variant}" / "summary.json")["seeds"]]
            value = {"published": False, "stable_features_min": min(stable), "stable_features_max": max(stable)}
        ranges.append(value)
    production = pd.concat([production, pd.DataFrame(ranges)], axis=1)
    production.to_csv(destination / "production.csv", index=False)
    figure(comparison, chosen, destination)
    write_json(destination / "provenance.json", {"campaign": config["campaign"], "nodule_coverage": config["nodule_coverage"], "selection_sha256": sha256(root / "selection.json"), "viewer_index_sha256": sha256(root / "viewer/index.json"),
                                                 "coverage_sha256": {path.name: sha256(path) for path in sorted((root / "coverage").glob("*.json"))}, "configuration_sha256": sha256(ROOT / "configs/sae_interpretation.json"),
                                                 "source_sha256": sha256(Path(__file__)), "python": platform.python_version(), "matplotlib": matplotlib.__version__, "created_at": timestamp(), "test_used": False})
    print("EXPORTED", destination, flush=True)


if __name__ == "__main__":
    main()

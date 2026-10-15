import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


SITES = [("fmcib", "layer1_context", "FMCIB layer1 · complete context"), ("fmcib", "layer2", "FMCIB layer2 · deployed window"), ("vista", "stage2", "VISTA3D stage2 · deployed window"), ("vista", "stage3", "VISTA3D stage3 · deployed window")]
COLORS = {"trained": "#2a78d6", "random": "#eb6834"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def value(rows, encoder, expansion, k, column):
    match = rows[(rows.encoder == encoder) & (rows.expansion == expansion) & (rows.k == k)]
    return match[column].item() if len(match) and column in match and pd.notna(match[column].item()) else np.nan


def figure(table, choices, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.8), facecolor=SURFACE, sharex=False)
    for column, (model, site, title) in enumerate(SITES):
        rows = table[(table.model == model) & (table.site == site)]
        configs = rows.query("encoder == 'trained'").sort_values(["expansion", "k"])[["expansion", "k"]].drop_duplicates().to_records(index=False)
        chosen = choices[f"{model}:{site}"]
        factor = chosen["training_steps"] // 12000
        labels = [f"e{e}\nk{k}" for e, k in configs] + [f"e{chosen['expansion']} k{chosen['k']}\n{factor}× tokens"]
        top_labels = [f"e{e}\nk{k}\n{value(rows, 'trained', e, k, 'stable_features'):.0f}" for e, k in configs] + [f"e{chosen['expansion']} k{chosen['k']}\n{factor}× tokens\n{chosen['stable_features']} / {chosen['control_stable_features']}"]
        positions = np.arange(len(labels), dtype=float)
        positions[-1] += 0.5
        top, bottom = axes[0, column], axes[1, column]
        grid = [value(rows, "trained", e, k, "stable_explained_variance") for e, k in configs]
        top.bar(positions[:-1], grid, width=0.5, color=COLORS["trained"], label="trained encoder", edgecolor=SURFACE, linewidth=1)
        top.bar(positions[-1] - 0.2, chosen["stable_explained_variance"], width=0.38, color=COLORS["trained"], edgecolor=SURFACE, linewidth=1)
        top.bar(positions[-1] + 0.2, chosen["control_stable_explained_variance"], width=0.38, color=COLORS["random"], label="random-weight encoder", edgecolor=SURFACE, linewidth=1)
        top.annotate("selected", (positions[-1], max(chosen["stable_explained_variance"], chosen["control_stable_explained_variance"], 0)), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=INK)
        top.axvline(positions[-1] - 0.75, color="#e4e3df", linewidth=1)
        top.axhline(0, color=MUTED, linewidth=0.8)
        top.set_title(title, fontsize=10, loc="left")
        top.set_xticks(positions, top_labels)
        mae = [value(rows, "trained", e, k, "probability_mae") for e, k in configs] + [chosen["probability_mae"]]
        bottom.bar(positions, mae, width=0.5, color=COLORS["trained"], edgecolor=SURFACE, linewidth=1)
        bottom.axvline(positions[-1] - 0.75, color="#e4e3df", linewidth=1)
        bottom.axhline(0.025, color=MUTED, linewidth=1, linestyle="--")
        bottom.text(positions[-1] + 0.4, 0.025, "gate 0.025", ha="right", va="bottom", fontsize=8, color=MUTED)
        bottom.set_xticks(positions, labels)
        for axis in [top, bottom]:
            axis.set_facecolor(SURFACE)
            axis.grid(axis="y", color="#e4e3df", linewidth=0.6)
            axis.set_axisbelow(True)
            for side in ["top", "right"]:
                axis.spines[side].set_visible(False)
        if column == 0:
            top.set_ylabel("Development variance explained\nby stable features (seed 2025)")
            top.text(-0.02, -0.02, "stable\nfeatures", transform=top.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
            bottom.set_ylabel("Fixed-classifier probability MAE\n(full replacement, seed 2025)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Interpretation dictionaries: 12,000-step grid (left of divider) and selected configuration with longer training (right)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"selection.{suffix}", dpi=180, facecolor=SURFACE)
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
    selection = read_json(root / "selection.json")
    table = pd.DataFrame(selection["configurations"])
    columns = ["encoder", "model", "site", "expansion", "k", "variant", "training_steps", "seeds", "features", "development_fvu", "development_cosine", "inactive_fraction", "l0_mean", "live_features", "stable_features", "unstable_features", "stable_fraction", "mean_stable_fraction_all_seeds", "stable_explained_variance", "sample_fvu", "pca_fvu", "pw_mcc", "unstable_subspace_energy", "unstable_subspace_chance", "probability_mae", "original_auroc", "recovered_auroc", "original_ap", "recovered_ap", "classifier_pass", "interior_probability_mae"]
    table.reindex(columns=columns).to_csv(destination / "configurations.csv", index=False)
    pd.DataFrame(selection["budget_diagnostic"]).reindex(columns=columns).assign(stable_rule="reappears in all 3 other seeds of 2025-2028").to_csv(destination / "training_length.csv", index=False)
    pd.DataFrame([choice["trained"] for choice in selection["choices"].values()] + [choice["control"] for choice in selection["choices"].values()]).reindex(columns=columns).to_csv(destination / "selected_configurations.csv", index=False)
    index = read_json(root / "viewer/index.json")
    selected = []
    for entry in index["dictionaries"]:
        row = {"encoder": entry["encoder"], "model": entry["model"], "site": entry["site"], "expansion": entry["expansion"], "k": entry["k"], "features": entry["features"], "seed": entry["seed"], "development_fvu": entry["quality"]["fvu"], "stable_features": entry["quality"]["stable_features"], "live_features": entry["quality"]["live_features"], "stable_explained_variance": entry["quality"]["stable_explained_variance"], "pca_fvu_reference": entry["quality"]["pca_fvu"], "pw_mcc_reference_pair": entry["quality"]["pw_mcc"]}
        if "downstream" in entry:
            row.update({"probability_mae": entry["downstream"]["probability_mae"], "original_auroc": entry["downstream"]["original_auroc"], "recovered_auroc": entry["downstream"]["recovered_auroc"], "original_ap": entry["downstream"]["original_ap"], "recovered_ap": entry["downstream"]["recovered_ap"]})
        if "interior_downstream" in entry:
            row["interior_probability_mae"] = entry["interior_downstream"]["probability_mae"]
        selected.append(row)
    pd.DataFrame(selected).to_csv(destination / "selected.csv", index=False)
    distribution = []
    for choice in selection["choices"].values():
        for encoder, group in [("trained", choice["group"]), ("random", choice["control_group"])]:
            with np.load(root / "stability" / group / "features.npz") as saved:
                counts = np.bincount(saved["reappearance"][0][saved["development_live"][0]], minlength=len(config["grid"]["seeds"]))
            distribution.append({"encoder": encoder, "model": choice["model"], "site": choice["site"], "expansion": choice["expansion"], "k": choice["k"], **{f"reappear_{index}_of_{len(counts) - 1}": int(value) for index, value in enumerate(counts)}})
    pd.DataFrame(distribution).to_csv(destination / "reappearance_distribution.csv", index=False)
    references = pd.read_csv(root / "nodule_references.csv")
    profiles = {f"{model}:{site}": read_json(root / "profiles" / f"{model}_{site}" / "summary.json")["size_only_development_auroc"] for model, site, _ in SITES}
    write_json(destination / "nodule_reference_summary.json", {"records": int(len(references)), "records_with_multiple_mask_components": int((references.mask_components > 1).sum()), "volume_mm3_quartiles": references.volume_mm3.quantile([0.25, 0.5, 0.75]).round(2).tolist(), "size_only_development_auroc_one_record_per_patient": profiles, "test_used": False})
    figure(table, selection["choices"], destination)
    write_json(destination / "provenance.json", {"campaign": config["campaign"], "selection_sha256": sha256(root / "selection.json"), "viewer_index_sha256": sha256(root / "viewer/index.json"), "configuration_sha256": sha256(ROOT / "configs/sae_interpretation.json"), "source_sha256": sha256(Path(__file__)), "python": platform.python_version(), "matplotlib": matplotlib.__version__, "created_at": timestamp(), "test_used": False})
    print("EXPORTED", destination, flush=True)


if __name__ == "__main__":
    main()

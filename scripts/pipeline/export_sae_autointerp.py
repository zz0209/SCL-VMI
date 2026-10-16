import argparse
import platform
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SITES = {"fmcib:layer2": "FMCIB layer2", "vista:stage2": "VISTA3D stage2"}
ROLE_ORDER = ["candidate", "nodule", "random"]


def short(key):
    _, _, role, feature = key.split("_", 3)
    return f"{role} {feature}"


def specificity_figure(pairs, features, destination):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), facecolor=SURFACE)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list("diverging", ["#2a78d6", "#f2f1ee", "#eb6834"])
    for axis, (site, title) in zip(axes, SITES.items()):
        part = pairs[pairs.site == site]
        order = sorted(part.target_key.unique(), key=lambda key: (ROLE_ORDER.index(features[key]["role"]), features[key]["feature"]))
        labels = [key for key in order if key in set(part.label_key)]
        matrix = part.pivot(index="label_key", columns="target_key", values="spearman").loc[labels, order]
        image = axis.imshow(matrix.to_numpy(), cmap=colormap, vmin=-0.75, vmax=0.75)
        for row, label in enumerate(labels):
            for column, target in enumerate(order):
                value = matrix.loc[label, target]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=INK, fontweight="bold" if label == target else "normal")
            column = order.index(label)
            axis.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
        axis.set_xticks(range(len(order)), [short(key) for key in order], rotation=50, ha="right")
        axis.set_yticks(range(len(labels)), [short(key) for key in labels])
        axis.set_xlabel("feature whose development case value is ranked")
        if site == "fmcib:layer2":
            axis.set_ylabel("label (direct route) used for scoring")
        axis.set_title(f"{title}: Spearman between label score and feature value, 60 development patients", fontsize=9, loc="left")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="Spearman correlation (outlined: the label's own feature)")
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"specificity.{suffix}", dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sae_autointerp.json")
    args = parser.parse_args()
    config = read_json(args.config)
    run = Path(context()[0]["runs"]) / config["run_id"]
    destination = args.output
    destination.mkdir(parents=True, exist_ok=False)
    features = {item["key"]: item for item in read_json(run / "features.json")["features"]}
    validation = pd.read_csv(run / "results" / "validation.csv")
    columns = ["site", "encoder", "role", "feature", "route", "no_clear_pattern", "candidate_label", "description", "auroc", "auroc_low", "auroc_high", "spearman", "mean_score_high", "mean_score_low",
               "measurement_auroc", "nodule_enrichment", "delta_auroc_volume"]
    validation[columns].to_csv(destination / "validation.csv", index=False)
    pd.read_csv(run / "results" / "summary_by_role.csv").to_csv(destination / "summary_by_role.csv", index=False)
    pairs = pd.read_csv(run / "results" / "specificity_pairs.csv")
    pairs.assign(label_feature=pairs.label_key.map(lambda key: features[key]["feature"]), label_role=pairs.label_key.map(lambda key: features[key]["role"]),
                 target_feature=pairs.target_key.map(lambda key: features[key]["feature"]), target_role=pairs.target_key.map(lambda key: features[key]["role"])).drop(columns=["label_key", "target_key"]).to_csv(destination / "specificity.csv", index=False)
    specificity_figure(pairs, features, destination)
    for suffix in ["png", "svg"]:
        shutil.copyfile(run / "results" / f"validation.{suffix}", destination / f"validation.{suffix}")
    summary = read_json(run / "results" / "summary.json")
    write_json(destination / "provenance.json", {"run_id": config["run_id"], "campaign": config["campaign"], "candidates_run": config["candidates_run"], "model": config["model"]["name"],
                                                 "settings": {key: config[key] for key in ["sites", "reference_seed", "near_mm", "features", "examples", "render", "specificity", "prompts"]},
                                                 "reasoning": config["model"]["reasoning"], "calls": summary["calls"], "total_cost_usd": summary["total_cost_usd"], "cost_by_stage": summary["cost_by_stage"],
                                                 "specificity": summary["specificity"], "matching_mean_abs_difference": read_json(run / "features.json")["matching_mean_abs_difference"],
                                                 "configuration": str(args.config), "configuration_sha256": sha256(args.config), "source_sha256": {name: sha256(ROOT / "src/sclvmi" / name) for name in ["sae_autointerp.py", "sae_autointerp_model.py"]},
                                                 "exporter_sha256": sha256(Path(__file__)), "python": platform.python_version(), "matplotlib": matplotlib.__version__, "created_at": timestamp(), "test_used": False})
    print("EXPORTED", destination, flush=True)


if __name__ == "__main__":
    main()

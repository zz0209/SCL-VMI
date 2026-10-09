import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from sclvmi.context import ROOT, context, read_json, timestamp, write_json


parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True)
parser.add_argument("--diagnostics", nargs="+", required=True)
parser.add_argument("--full-comparisons", nargs="+", default=[])
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
storage, _, _ = context()
root = Path(storage["runs"]) / "20261008_spatial_sae"
args.output.mkdir(parents=True, exist_ok=True)
rows = []
index = []
colors = {2025: "#0072B2", 2026: "#D55E00", 2027: "#009E73"}
fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for run_id in args.runs:
    run = root / run_id
    request = read_json(run / "request.json")
    result = read_json(run / "result.json")
    downstream = read_json(run / "downstream_full/result.json")
    bootstrap = read_json(run / "downstream_full/bootstrap.json")
    inference = read_json(run / "inference_checks.json")
    assert result["feature_checks_pass"] and downstream["checks_pass"]
    assert inference["status"] == "passed" and inference["dictionary_sha256"] == result["dictionary_sha256"]
    assert not result["smoke"] and not result["test_used"]
    row = {"run_id": run_id, "model": request["model"], "site": request["site"], "seed": request["seed"], "dictionary_size": request["dict_size"], "target_l0": request["k"], "steps": result["steps"], "best_step": result["best_step"], **{key: result["development"][key] for key in ["fvu", "cosine", "l0_mean", "inactive_fraction"]}, **downstream["metrics"]}
    for name, value in zip(["median", "p90", "p99", "max"], row.pop("probability_error_quantiles")):
        row[f"probability_error_{name}"] = value
    for key, interval in bootstrap["intervals"].items():
        row[f"{key}_ci_lower"], row[f"{key}_ci_upper"] = interval
    rows.append(row)
    index.append({"model": request["model"], "site": request["site"], "seed": request["seed"], "run_id": run_id, "dictionary": str(run / "dictionary.pt"), "dictionary_sha256": result["dictionary_sha256"], "input_dim": request["input_dim"], "dictionary_size": request["dict_size"], "target_l0": request["k"], "default": request["seed"] == 2025, "numerical_checks_pass": True})
    history = pd.DataFrame(read_json(run / "history.json"))
    model_index = 0 if request["model"] == "vista" else 1
    for column, metric in enumerate(["fvu", "l0_mean"]):
        axes[model_index, column].plot(history.step, history[metric], color=colors[request["seed"]], label=f"Seed {request['seed']}", linewidth=1.7)
        axes[model_index, column].set_xlabel("Training step")
        axes[model_index, column].set_ylabel("Unexplained variance (FVU)" if metric == "fvu" else "Mean active features")
        axes[model_index, column].set_title(f"{request['model'].upper()} / {request['site']}")
        axes[model_index, column].grid(alpha=0.18)
        axes[model_index, column].spines[["top", "right"]].set_visible(False)
    axes[model_index, 0].axhline(request["acceptance"]["max_development_fvu"], color="#666666", linewidth=0.7, linestyle=":")
for ax in axes.flat:
    ax.legend(frameon=False, fontsize=8)
fig.savefig(args.output / "training_quality.png", dpi=200)
fig.savefig(args.output / "training_quality.svg")
plt.close(fig)
pd.DataFrame(rows).to_csv(args.output / "selected.csv", index=False)
candidates = []
for request_path in sorted(root.glob("*/request.json")):
    request = read_json(request_path)
    if request["smoke"]:
        continue
    result_path = request_path.parent / "result.json"
    if not result_path.exists():
        continue
    result = read_json(result_path)
    row = {"run_id": request["run_id"], "model": request["model"], "site": request["site"], "seed": request["seed"], "steps": result["steps"], "expansion": request["expansion"], "dictionary_size": request["dict_size"], "target_l0": request["k"], "best_step": result["best_step"], **{key: result["development"][key] for key in ["fvu", "cosine", "l0_mean", "inactive_fraction"]}, "feature_checks_pass": result["feature_checks_pass"], "selected": request["run_id"] in args.runs, "downstream_scope": "not_evaluated"}
    for scope in ["full", "patients64"]:
        downstream_path = request_path.parent / f"downstream_{scope}/result.json"
        if downstream_path.exists():
            downstream = read_json(downstream_path)
            row.update({"downstream_scope": scope, "downstream_checks_pass": downstream["checks_pass"], **{key: downstream["metrics"][key] for key in ["cases", "patients", "probability_mae", "original_auroc", "recovered_auroc", "original_ap", "recovered_ap"]}})
            break
    candidates.append(row)
comparison = pd.DataFrame(candidates)
comparison.to_csv(args.output / "candidate_comparison.csv", index=False)
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
candidate_metrics = ["fvu", "cosine", "inactive_fraction"]
metric_labels = {"fvu": "Unexplained variance (FVU)", "cosine": "Mean token cosine", "inactive_fraction": "Inactive development features"}
targets = {"fvu": 0.15, "cosine": 0.95, "inactive_fraction": 0.02}
for model_index, name in enumerate(["vista", "fmcib"]):
    subset = comparison.query("model == @name and seed == 2025 and steps == 12000")
    shorter = comparison.query("model == @name and seed == 2025 and steps != 12000 and selected")
    for (site, expansion), group in subset.groupby(["site", "expansion"], sort=True):
        group = group.sort_values("target_l0")
        for column, metric in enumerate(candidate_metrics):
            axes[model_index, column].plot(group.target_l0, group[metric], marker="o", linewidth=1.5, label=f"{site}, expansion {expansion}")
    for column, metric in enumerate(candidate_metrics):
        ax = axes[model_index, column]
        ax.set_title(f"{name.upper()} / 12,000 steps / seed 2025")
        ax.set_xlabel("Target mean active features")
        ax.set_ylabel(metric_labels[metric])
        if metric == "inactive_fraction":
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.axhline(targets[metric], color="#666666", linewidth=0.8, linestyle=":", label="Acceptance target")
        chosen = subset[subset.selected]
        ax.scatter(chosen.target_l0, chosen[metric], s=110, facecolors="none", edgecolors="black", linewidths=1.3, zorder=5)
        for row in shorter.itertuples():
            ax.scatter(row.target_l0, getattr(row, metric), marker="*", s=160, color="#CC79A7", edgecolors="black", linewidths=0.6, zorder=6, label=f"Selected: {row.site}, {row.steps:,} steps")
        if len(shorter):
            ax.set_title(f"{name.upper()} / seed 2025\nLines: 12,000 steps; star: selected shorter budget")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
fig.savefig(args.output / "candidate_quality.png", dpi=200)
fig.savefig(args.output / "candidate_quality.svg")
plt.close(fig)
diagnostics = {}
for name in args.diagnostics:
    directory = root / name
    pca = read_json(directory / "pca_reference.json")
    seeds = read_json(directory / "seed_comparison.json")
    assert set(seeds["runs"]).issubset(args.runs)
    diagnostics[name] = {"pca_reference": pca, "seed_comparison": seeds}
for name in args.full_comparisons:
    seeds = read_json(root / name / "seed_comparison.json")
    assert set(seeds["runs"]).issubset(args.runs)
    diagnostics[name] = {"full_seed_comparison": seeds}
write_json(args.output / "diagnostics.json", diagnostics)
write_json(root / "dictionary_index.json", {"status": "numerically_validated", "scope": "spatial SAE preparation for later crop/heatmap and semantic inspection", "dictionaries": index, "test_used": False, "created_at": timestamp()})
print(json.dumps({"dictionaries": len(index), "report": str(args.output), "index": str(root / "dictionary_index.json")}), flush=True)

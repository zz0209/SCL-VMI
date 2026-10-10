import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sclvmi.context import read_json, sha256, timestamp, write_json
from sclvmi.sae_campaign import campaign_root, configuration


def collect(root):
    rows = []
    sources = []
    for job in read_json(root / "jobs.json")["jobs"]:
        run = root / "runs" / job["run_id"]
        path = run / "result.json"
        record = {key: job[key] for key in ["run_id", "model", "site", "expansion", "k", "seed", "phase"]}
        record["enabled"] = job.get("enabled", True)
        record["algorithm"] = job.get("training", {}).get("sae_type", "batch_topk")
        record["checkpoint_policy"] = job.get("training", {}).get("checkpoint_selection", "fvu")
        record["state"] = "pending" if record["enabled"] else "superseded"
        record["classifier_state"] = "pending"
        if (run / "failure.json").exists():
            record["state"] = "failed"
        if path.exists():
            result = read_json(path)
            record.update(state="completed", best_step=result["best_step"], features=result["request"]["dict_size"], representation_pass=result["feature_checks_pass"])
            record.update({key: result["development"][key] for key in ["fvu", "cosine", "l0_mean", "inactive_fraction", "patients", "tokens"]})
            sources.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path)})
            downstream = run / "downstream.json"
            if downstream.exists():
                evaluation = read_json(downstream)
                record.update(classifier_state="completed", classifier_pass=evaluation["checks_pass"])
                record.update(evaluation["metrics"])
                sources.append({"path": downstream.relative_to(root).as_posix(), "sha256": sha256(downstream)})
            elif not result["feature_checks_pass"]:
                record["classifier_state"] = "not_required_representation_failed"
        rows.append(record)
    return pd.DataFrame(rows), sources


def draw_candidates(frame, output):
    completed = frame.query("state == 'completed' and seed == 2025")
    sites = [(model, site) for model, values in configuration()["models"].items() for site in values]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), layout="constrained", sharey=True, sharex=True)
        maximum_fvu = max(.15, float(completed.fvu.max())) * 1.12
        maximum_activity = max(1, float(completed.l0_mean.max())) * 1.08
        for axis, (model, site) in zip(axes.flat, sites):
            values = completed.loc[(completed.model == model) & (completed.site == site)]
            for _, row in values.iterrows():
                qualified = row.representation_pass and pd.notna(row.get("classifier_pass")) and bool(row.get("classifier_pass"))
                marker = "s" if row.algorithm == "topk" else "o"
                color = "#0072B2" if qualified else "#666666"
                axis.scatter(row.l0_mean, row.fvu, s=52, marker=marker, edgecolors=color, facecolors=color if qualified else "none", linewidths=1.4)
            axis.axhline(configuration()["acceptance"]["max_development_fvu"], color="#222222", linestyle="--", linewidth=.9)
            axis.set(title=f"{model.upper()} · {site}", xlabel="Mean active features", ylabel="Development FVU", xlim=(0, maximum_activity), ylim=(0, maximum_fvu))
            if values.empty:
                axis.text(.5, .5, "Results pending", ha="center", va="center", transform=axis.transAxes, color="#555555")
            axis.grid(alpha=.18)
        fig.suptitle("SAE candidate reconstruction and activity · seed 2025", fontsize=16)
        fig.supxlabel("Circles: BatchTopK · Squares: TopK · Filled blue: all quality checks passed · Dashed line: FVU criterion", fontsize=10)
        for extension in ["png", "svg"]:
            fig.savefig(output / f"candidate_quality.{extension}", dpi=180, facecolor="white")
        plt.close(fig)


def export_selection(root, frame, output, sources):
    selection_path = root / "selection.json"
    selection = read_json(selection_path)
    assert selection["state"] == "completed"
    sources.append({"path": "selection.json", "sha256": sha256(selection_path)})
    chosen, comparisons, references = [], [], []
    for entry in selection["selected"]:
        model, site = entry["model"], entry["site"]
        group = f"{model.upper()} {site}"
        selected = frame.loc[frame.run_id.isin(entry["replicas"])].copy()
        assert len(selected) == len(entry["replicas"]) == 3
        assert selected.representation_pass.all() and selected.classifier_pass.all()
        selected["role"] = entry["role"]
        selected["group"] = group
        chosen.append(selected)
        diagnostic_path = root / entry["diagnostics"]
        diagnostic = read_json(diagnostic_path)
        assert set(diagnostic["runs"]) == set(entry["replicas"])
        sources.append({"path": diagnostic_path.relative_to(root).as_posix(), "sha256": sha256(diagnostic_path)})
        for pair in diagnostic["comparisons"]:
            quantiles = pair["correlation_quantiles"]
            comparisons.append({"group": group, "role": entry["role"], "pair": pair["pair"], "q10": quantiles[1], "median": quantiles[2], "q90": quantiles[3], "source_eligible_features": pair["source_eligible_features"], "source_features": pair["source_features"], "target_eligible_features": pair["target_eligible_features"], "reciprocal_fraction": pair["reciprocal_fraction"], "reciprocal_and_correlation_ge_0_8": pair["reciprocal_and_correlation_ge_0_8"]})
        pca_path = root / "runs" / entry["replicas"][0] / "pca_reference/result.json"
        pca = read_json(pca_path)
        references.append({"group": group, "rank": pca["rank"], "development_fvu": pca["development_fvu"]})
        sources.append({"path": pca_path.relative_to(root).as_posix(), "sha256": sha256(pca_path)})
    selected = pd.concat(chosen, ignore_index=True)
    stability = pd.DataFrame(comparisons)
    selected.to_csv(output / "selected.csv", index=False, na_rep="")
    stability.to_csv(output / "seed_consistency.csv", index=False)
    pd.DataFrame(references).to_csv(output / "pca_reference.csv", index=False)
    groups = selected.group.drop_duplicates().tolist()
    colors = ["#0072B2", "#D55E00", "#009E73"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(1, 3, figsize=(15, max(5, len(groups) * .75)), layout="constrained", sharey=True)
        for index, group in enumerate(groups):
            values = selected.loc[selected.group == group].sort_values("seed")
            for offset, (_, row), color in zip([-.18, 0, .18], values.iterrows(), colors):
                axes[0].scatter(row.fvu, index + offset, color=color, s=42, label=str(row.seed) if index == 0 else None)
                axes[1].scatter(row.probability_mae, index + offset, color=color, s=42)
            reference = next(row for row in references if row["group"] == group)
            axes[0].scatter(reference["development_fvu"], index, marker="x", color="#222222", s=60, label="PCA" if index == 0 else None)
            for offset, (_, pair), color in zip([-.18, 0, .18], stability.loc[stability.group == group].iterrows(), colors):
                axes[2].plot([pair.q10, pair.q90], [index + offset] * 2, color=color, linewidth=1.8, label=pair["pair"].replace("s", "").replace("_", " / ") if index == 0 else None)
                axes[2].scatter(pair["median"], index + offset, color=color, s=35)
        axes[0].set_yticks(range(len(groups)), groups)
        axes[0].invert_yaxis()
        axes[0].set(xlabel="Development FVU", title="Representation reconstruction", xlim=(0, None))
        axes[1].set(xlabel="Mean absolute probability change", title="Fixed classifier", xlim=(0, None))
        axes[2].set(xlabel="Matched activation correlation", title="Cross-seed feature consistency", xlim=(min(0, float(stability.q10.min()) - .02), 1.02))
        axes[0].axvline(configuration()["acceptance"]["max_development_fvu"], linestyle="--", color="#777777", linewidth=.8)
        axes[1].axvline(configuration()["acceptance"]["max_probability_mae"], linestyle="--", color="#777777", linewidth=.8)
        axes[0].legend(frameon=False, fontsize=9)
        axes[2].legend(frameon=False, fontsize=9, title="Seed pair")
        for axis in axes:
            axis.grid(axis="x", alpha=.18)
        fig.suptitle("Selected SAE configurations across three training seeds", fontsize=16)
        fig.supxlabel("Left and center: individual seeds. Right: three seed pairs; dots = median, lines = 10th–90th percentiles over nonconstant source features.", fontsize=9)
        for extension in ["png", "svg"]:
            fig.savefig(output / f"selected_quality.{extension}", dpi=180, facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", action="store_true")
    args = parser.parse_args()
    root = campaign_root(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    frame, sources = collect(root)
    assert frame.run_id.is_unique
    frame.to_csv(args.output / "candidates.csv", index=False, na_rep="")
    draw_candidates(frame, args.output)
    if args.selection:
        export_selection(root, frame, args.output, sources)
    write_json(args.output / "provenance.json", {"created_at": timestamp(), "source_code_sha256": sha256(Path(__file__)), "sources": sources, "matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__, "unit": "One training run; development metrics weight patients equally", "transformations": "Direct saved aggregate metrics; no smoothing or fitted trends; all completed seed-2025 candidates appear in the candidate figure", "missing_values": "Empty CSV cells are uncomputed; classifier_state distinguishes pending and representation-gated checks", "uncertainty": "Training metrics show individual seeds without confidence intervals. Seed-consistency lines span the 10th to 90th percentiles of feature correlations, conditional on nonconstant source features; these are distribution intervals, not confidence intervals.", "pca_comparison": "Training-fitted patient-weighted rank-k PCA uses the target active-coefficient budget. BatchTopK realized activity is separately reported; dictionary storage and representation families differ.", "test_used": False})
    print({"output": str(args.output), "runs": len(frame), "completed": int(frame.state.eq("completed").sum())}, flush=True)


if __name__ == "__main__":
    main()

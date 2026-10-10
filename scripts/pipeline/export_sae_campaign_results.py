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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = campaign_root(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    frame, sources = collect(root)
    assert frame.run_id.is_unique
    frame.to_csv(args.output / "candidates.csv", index=False, na_rep="")
    draw_candidates(frame, args.output)
    write_json(args.output / "provenance.json", {"created_at": timestamp(), "source_code_sha256": sha256(Path(__file__)), "sources": sources, "matplotlib": matplotlib.__version__, "pandas": pd.__version__, "numpy": np.__version__, "unit": "One training run; development metrics weight patients equally", "transformations": "Direct saved aggregate metrics; no smoothing or fitted trends; all completed seed-2025 candidates appear in the figure", "missing_values": "Empty CSV cells are uncomputed; classifier_state distinguishes pending and representation-gated checks", "uncertainty": "Points are observed run metrics without uncertainty intervals", "test_used": False})
    print({"output": str(args.output), "runs": len(frame), "completed": int(frame.state.eq("completed").sum())}, flush=True)


if __name__ == "__main__":
    main()

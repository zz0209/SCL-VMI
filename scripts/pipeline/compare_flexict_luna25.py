import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.head import metrics


SPEC = read_json(ROOT / "configs/flexict_luna25.json")
REMOTE_RUNS = f"{read_json(ROOT / 'configs/sae_remote.local.json')['remote_root']}/{SPEC['remote_directory']}/runs/{SPEC['suite']}"
BASELINE = "20261017_encoder_pipeline"
LAYER_SUITES = ["20261017_flexict_improvement", "20261017_flexict_improvement_extension"]
AUTOMSC_RUN = "20261017_automsc_mayo_development_s2025"
VIEWS = {"f32": "32 mm field", "f40": "40 mm field", "c80": "80 mm field, central 40 mm", "w15": "whole CT 1.5 mm", "w20": "whole CT 2.0 mm (official CT-RATE form)"}
MODELS = {"flexict_3d": "FlexiCT-3D", "flexict_vlm": "FlexiCT-3D-VLM image encoder"}
METRICS = ["auroc", "average_precision", "brier", "log_loss"]


def runs_root():
    return Path(context()[0]["runs"])


def local_run(path):
    assert path.startswith(REMOTE_RUNS), path
    return runs_root() / SPEC["suite"] / "mayo" / path.removeprefix(REMOTE_RUNS + "/")


def describe(label):
    if label.startswith("fusion_"):
        parts = label.removeprefix("fusion_").split("_")
        return f"{VIEWS[parts[0]]} block {int(parts[1][1:])} + {VIEWS[parts[2]]} block {int(parts[3][1:])} with its class token, max head"
    view, block = label.split("_b")
    return f"{VIEWS[view]}, block {int(block)} of 16, max head"


def load(paths):
    frames = [pd.read_csv(path, dtype={"PatientID": str}).sort_values("identifier").reset_index(drop=True) for path in paths]
    for frame in frames:
        assert frame.identifier.is_unique and frame.probability.between(0, 1).all()
    return frames


def reproduction():
    # seed-2025 development AUROC of the provider-crop fields rebuilt from the whole CTs against the baseline suites
    old = pd.concat([pd.read_csv(runs_root() / suite / "heads" / name / "candidates.csv") for suite in LAYER_SUITES for name in MODELS])
    old = old[(old.seed == 2025) & old.candidate.str.match(r"^f(32|40)_b\d+$")].groupby(["model", "candidate", "lr"]).auroc.first()
    rows = []
    for name in MODELS:
        new = pd.read_csv(runs_root() / SPEC["suite"] / "mayo" / "heads" / name / "candidates.csv")
        new = new[(new.seed == 2025) & new.candidate.str.match(r"^f(32|40)_b\d+$")]
        for row in new.itertuples():
            if (name, row.candidate, row.lr) in old.index:
                rows.append({"model": name, "candidate": row.candidate, "lr": row.lr, "baseline_auroc": old[(name, row.candidate, row.lr)], "rebuilt_auroc": row.auroc})
    table = pd.DataFrame(rows)
    table["difference"] = table.rebuilt_auroc - table.baseline_auroc
    return table


def main(public):
    output = runs_root() / SPEC["suite"] / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    baseline = runs_root() / BASELINE / "comparison"
    provenance = read_json(baseline / "provenance.json")
    previous = pd.read_csv(baseline / "comparison.csv").set_index("key")
    paths = {}
    for key, group in provenance["predictions"].items():
        for item in group:
            assert sha256(item["path"]) == item["sha256"], item["path"]
        paths[key] = [Path(item["path"]) for item in group]
    labels = {key: previous.loc[key, "model"] for key in paths}
    pipelines = {key: previous.loc[key, "pipeline"] for key in paths}
    automsc = runs_root() / AUTOMSC_RUN
    if (automsc / "status.json").exists() and read_json(automsc / "status.json")["status"] == "completed":
        paths["automsc"], labels["automsc"], pipelines["automsc"] = [automsc / "predictions.csv"], "AutoMSC (classification + segmentation)", "official AutoMSC training"
    summary = read_json(runs_root() / SPEC["suite"] / "mayo" / "report" / "summary.json")
    families = {}
    for name, item in summary["models"].items():
        selection = read_json(runs_root() / SPEC["suite"] / "mayo" / "heads" / name / "selection.json")
        for role in ["best", "best_crop", "best_context"]:
            label = item[role]
            key = f"{name}_luna25_{label}"
            if key not in families:
                families[key] = {"model": name, "label": label, "roles": [], "runs": [local_run(run) for run in selection["runs"][label]]}
            families[key]["roles"].append(role)
    for key, family in families.items():
        paths[key] = [run / "development_predictions.csv" for run in family["runs"]]
        labels[key] = f"{MODELS[family['model']]}, official LUNA25 CTs"
        pipelines[key] = describe(family["label"])
    frames = {key: load(group) for key, group in paths.items()}
    reference = frames["i3d"][0][["identifier", "PatientID", "label"]]
    assert len(reference) == 1285 and reference.PatientID.nunique() == 422
    for group in frames.values():
        for frame in group:
            pd.testing.assert_frame_equal(reference, frame[reference.columns])
    names = list(paths)
    truth = reference.label.to_numpy()
    groups = list(reference.groupby("PatientID", sort=True).indices.values())
    probabilities = {name: [frame.probability.to_numpy() for frame in frames[name]] for name in names}
    generator = np.random.default_rng(2025)
    draws = []
    for _ in tqdm(range(2000), desc="Paired patient bootstrap", mininterval=3):
        indices = np.concatenate([groups[i] for i in generator.integers(0, len(groups), len(groups))])
        draws.append([[np.mean([function(truth[indices], p[indices]) for p in probabilities[name]]) for function in (roc_auc_score, average_precision_score)] for name in names])
    draws = np.asarray(draws)
    rows = []
    for index, name in enumerate(names):
        scores = [metrics(truth, p) for p in probabilities[name]]
        row = {"model": labels[name], "key": name, "group": "official LUNA25 CTs" if name in families else "provider crops", "pipeline": pipelines[name], "seed_count": len(scores),
               "roles": ", ".join(families[name]["roles"]) if name in families else "", **{metric: float(np.mean([score[metric] for score in scores])) for metric in METRICS}}
        for position, metric in enumerate(["auroc", "average_precision"]):
            row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(draws[:, index, position], [0.025, 0.975])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    differences = []
    for key, family in families.items():
        for other, title in [(family["model"], f"{MODELS[family['model']]}, provider crops (selected baseline)"), ("vista", "VISTA3D"), ("i3d", "official LUNA25 I3D")]:
            low, high = np.quantile(draws[:, names.index(key), 0] - draws[:, names.index(other), 0], [0.025, 0.975])
            differences.append({"pipeline": f"{labels[key]}: {pipelines[key]}", "roles": ", ".join(family["roles"]), "minus": title,
                                "auroc_difference": rows[names.index(key)]["auroc"] - rows[names.index(other)]["auroc"], "ci_low": low, "ci_high": high})
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    repro = reproduction()
    repro.to_csv(output / "reproduction.csv", index=False)
    figure(table, output)
    record = {"created_at": timestamp(), "source_sha256": sha256(Path(__file__)), "config_sha256": sha256(ROOT / "configs/flexict_luna25.json"), "resamples": len(draws), "seed": 2025,
              "predictions": {name: [{"path": str(path), "sha256": sha256(path)} for path in group] for name, group in paths.items()}, "test_used": False,
              "context_gain": {name: item["context_gain"] for name, item in summary["models"].items()},
              "reproduction": {"pairs": len(repro), "max_abs_difference": float(repro.difference.abs().max()), "mean_difference": float(repro.difference.mean())},
              "aggregation": "mean of metrics across the three seeds of a selected configuration; configurations were selected on development data"}
    write_json(output / "provenance.json", record)
    if public:
        publish(output, public, table, record)
    print(table[["model", "pipeline", "roles", "seed_count", "auroc", "auroc_ci_low", "auroc_ci_high", "average_precision"]].sort_values("auroc", ascending=False).to_string(index=False), flush=True)
    print(pd.DataFrame(differences).to_string(index=False), flush=True)
    print("reproduction", record["reproduction"], flush=True)


def figure(table, output):
    ink, muted, surface, grid, crops, whole = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e2dd", "#8a8780", "#2a62c9"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": muted})
    ordered = table.sort_values("auroc").reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(8.6, 0.34 * len(ordered) + 1.2), facecolor=surface)
    axis.set_facecolor(surface)
    for position, row in ordered.iterrows():
        color = whole if row.group == "official LUNA25 CTs" else crops
        axis.plot([row.auroc_ci_low, row.auroc_ci_high], [position, position], color=color, linewidth=2, solid_capstyle="round")
        axis.plot(row.auroc, position, "o", color=color, markersize=6, markeredgecolor=surface, markeredgewidth=1.5)
        axis.text(row.auroc_ci_high + 0.004, position, f"{row.auroc:.3f}", va="center", fontsize=8, color=ink)
    axis.set_yticks(range(len(ordered)), [f"{row.model}: {row.pipeline}" for row in ordered.itertuples()], fontsize=7.5)
    axis.set_xlabel("development AUROC (mean of seeds; 95% patient-bootstrap interval)")
    axis.grid(axis="x", color=grid, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ["top", "right"]:
        axis.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], color=crops, marker="o", linewidth=2, label="inputs from provider crops"), plt.Line2D([], [], color=whole, marker="o", linewidth=2, label="FlexiCT on official LUNA25 CTs")]
    axis.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    axis.set_title("LUNA25 development fold (1,285 nodules, 422 patients)", loc="left", fontsize=9.5)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"comparison.{suffix}", dpi=170, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def publish(output, destination, table, record):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["comparison.csv", "paired_differences.csv", "reproduction.csv", "comparison.png", "comparison.svg"]:
        shutil.copyfile(output / name, destination / name)
    report = runs_root() / SPEC["suite"] / "mayo" / "report"
    pd.read_csv(report / "candidates.csv").drop(columns="run").to_csv(destination / "candidates.csv", index=False)
    for name in ["views_blocks.png", "views_blocks.svg"]:
        shutil.copyfile(report / name, destination / name)
    write_json(destination / "provenance.json", {key: value for key, value in record.items() if key != "predictions"} | {"prediction_sha256": {name: [item["sha256"] for item in group] for name, group in record["predictions"].items()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, help="results directory for the public export")
    main(parser.parse_args().public)

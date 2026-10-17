import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.encoder_features import suite_root
from sclvmi.head import metrics


EXISTING = {"fmcib": "FMCIB", "ctfm": "CT-FM", "vista": "VISTA3D", "genesis": "Models Genesis", "coralbay": "CoralBay", "tapct": "TAP-CT"}
NEW = {"nnfoundation_cnn": "nnFoundation CNN", "nnfoundation_cnn_native": "nnFoundation CNN, native crop", "nnfoundation_vit": "nnFoundation ViT", "flexict_3d": "FlexiCT-3D", "flexict_vlm": "FlexiCT-3D-VLM image encoder"}
FIELD_NAMES = {"nnfoundation_cnn_native": {"reference field": "last stage", "local field": "penultimate stage", "two fields": "three stages"}}
AUTOMSC_RUN = "20261017_automsc_mayo_development_s2025"
METRICS = ["auroc", "average_precision", "brier", "log_loss"]


LINEAR_VIEWS = {"native_linear": "reference", "spatial_linear": "reference", "reference_meanmax": "reference", "local_meanmax": "local", "local_multistage": "local", "two_view": "two views"}
HEADS = {"native_linear": "embedding linear", "spatial_linear": "spatial linear", "native_mlp": "embedding MLP", "mean": "mean head", "max": "max head", "attention": "attention head",
         "reference_meanmax": "mean+max linear", "local_meanmax": "mean+max linear", "local_multistage": "two-stage mean+max linear", "two_view": "two-field mean+max linear"}


def pipeline_label(name, kind, arm):
    field = {"reference": "reference field", "local": "local field", "local_max": "local field"}.get(kind) or {"reference": "reference field", "local": "local field", "two views": "two fields"}[LINEAR_VIEWS[arm]]
    head = "mean+max linear" if name in FIELD_NAMES and arm == "two_view" else HEADS[arm]
    return f"{FIELD_NAMES.get(name, {}).get(field, field)}, {head}"


def describe(run):
    result = read_json(run / "result.json")
    view = {"reference_heads": "reference", "local_heads": "local"}[run.parent.name] if "seed" in result else LINEAR_VIEWS[result["arm"]]
    return result, view


def select(name):
    runs = [Path(path) for path in read_json(suite_root() / "heads" / name / "candidates.json")["runs"]]
    families, rows = defaultdict(list), []
    for run in runs:
        result, view = describe(run)
        family = (run.parent.name, result["arm"], result.get("candidate"))
        families[family].append((run, result))
        rows.append({"model": name, "head": result["arm"], "view": view, "run_id": run.name, "seed": result.get("seed"), "lr": result.get("lr"), **{metric: result["metrics"][metric] for metric in METRICS}})
    choices = []
    for (_, arm, candidate), members in families.items():
        if candidate is not None and len(members) == 1:
            continue
        scores = {metric: float(np.mean([result["metrics"][metric] for _, result in members])) for metric in METRICS}
        choices.append({"family": members, "arm": arm, "metrics": scores, "auroc_sd": float(np.std([result["metrics"]["auroc"] for _, result in members]))})
    best = max(choices, key=lambda choice: (choice["metrics"]["auroc"], -choice["metrics"]["log_loss"]))
    run, result = best["family"][0]
    selected = {"run": str(run), "kind": describe(run)[1], "result": result, "repeat_runs": [str(member) for member, _ in best["family"]], "aggregate_metrics": best["metrics"], "auroc_sd": best["auroc_sd"],
                "selection_reason": "Highest mean development AUROC", "prediction_sha256": sha256(run / "development_predictions.csv")}
    return selected, pd.DataFrame(rows)


def predictions(paths):
    frames = {name: [pd.read_csv(path, dtype={"PatientID": str}).sort_values("identifier").reset_index(drop=True) for path in group] for name, group in paths.items()}
    reference = frames["i3d"][0][["identifier", "PatientID", "label"]]
    assert len(reference) == 1285 and reference.PatientID.nunique() == 422
    for group in frames.values():
        for table in group:
            assert table.identifier.is_unique and table.probability.between(0, 1).all()
            pd.testing.assert_frame_equal(reference, table[reference.columns])
    return frames


def main(public=None):
    runs = Path(context()[0]["runs"])
    output = suite_root() / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    existing = read_json(runs / "20260926_material_preparation/selected.json")
    selected, candidates = {}, []
    for name in NEW:
        if (suite_root() / f"{name}_complete.json").exists():
            selected[name], table = select(name)
            candidates.append(table)
    assert selected, "No completed new encoder"
    write_json(suite_root() / "selected.json", selected)
    pd.concat(candidates).to_csv(output / "candidates.csv", index=False)
    labels = {**EXISTING, **{name: NEW[name] for name in selected}, "i3d": "Official LUNA25 I3D"}
    paths = {name: [Path(run) / "development_predictions.csv" for run in existing[name]["repeat_runs"]] for name in EXISTING}
    paths.update({name: [Path(run) / "development_predictions.csv" for run in selected[name]["repeat_runs"]] for name in selected})
    paths["i3d"] = [runs / "20260925_official_i3d_s2025/development_predictions.csv"]
    automsc = runs / AUTOMSC_RUN
    if (automsc / "status.json").exists() and read_json(automsc / "status.json")["status"] == "completed":
        paths["automsc"], labels["automsc"] = [automsc / "predictions.csv"], "AutoMSC (classification + segmentation)"
    frames = predictions(paths)
    names = list(labels)
    truth = frames["i3d"][0].label.to_numpy()
    groups = list(frames["i3d"][0].groupby("PatientID", sort=True).indices.values())
    probabilities = {name: [frame.probability.to_numpy() for frame in frames[name]] for name in names}
    generator = np.random.default_rng(2025)
    draws = []
    for _ in tqdm(range(2000), desc="Paired patient bootstrap", mininterval=3):
        indices = np.concatenate([groups[i] for i in generator.integers(0, len(groups), len(groups))])
        draws.append([[np.mean([function(truth[indices], p[indices]) for p in probabilities[name]]) for function in (roc_auc_score, average_precision_score)] for name in names])
    draws = np.asarray(draws)
    np.save(output / "bootstrap.npy", draws)
    rows = []
    for index, name in enumerate(names):
        scores = [read_json(path.parent / "result.json")["metrics"] for path in paths[name]] if name in EXISTING or name in NEW else [metrics(truth, p) for p in probabilities[name]]
        source = selected[name] if name in selected else existing.get(name)
        row = {"model": labels[name], "key": name, "group": "new" if name in NEW else "existing" if name in EXISTING else "reference", "pipeline": pipeline_label(name, source["kind"], source["result"]["arm"]) if source else "official model", "seed_count": len(scores),
               **{metric: float(np.mean([score[metric] for score in scores])) for metric in METRICS}}
        for position, metric in enumerate(["auroc", "average_precision"]):
            row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(draws[:, index, position], [0.025, 0.975])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    differences = []
    for index, name in enumerate(names):
        if name in NEW:
            for other in ["vista", "i3d"]:
                position = names.index(other)
                low, high = np.quantile(draws[:, index, 0] - draws[:, position, 0], [0.025, 0.975])
                differences.append({"comparison": f"{labels[name]} minus {labels[other]}", "auroc_difference": rows[index]["auroc"] - rows[position]["auroc"], "ci_low": low, "ci_high": high})
    pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
    figure(table, output)
    provenance = {"created_at": timestamp(), "source_sha256": sha256(Path(__file__)), "specification_sha256": sha256(ROOT / "configs/encoder_pipeline.json"),
                  "predictions": {name: [{"path": str(path), "sha256": sha256(path)} for path in group] for name, group in paths.items()}, "resamples": len(draws), "seed": 2025,
                  "test_used": False, "aggregation": "mean of metrics across the selected configuration's seeds; configuration selection used development data"}
    write_json(output / "provenance.json", provenance)
    if public:
        publish(output, public, table, provenance)
    print(table[["model", "pipeline", "seed_count", "auroc", "auroc_ci_low", "auroc_ci_high", "average_precision"]].to_string(index=False), flush=True)


PUBLIC_README = """# nnFoundation and FlexiCT as frozen LUNA25 encoders

Suite `20261017_encoder_pipeline` evaluates nnFoundation (CNN and ViT; arXiv:2609.26924) and FlexiCT (the 3D self-supervised teacher and the image encoder of its 3D vision-language model; arXiv:2605.21906) as frozen encoders under the same procedure as the six existing encoders, and compares them with those encoders and the locally trained official LUNA25 I3D.

![Development comparison](comparison.png)

## Procedure

- **Data:** training uses 3,729 records from 1,264 patients and development uses 1,285 records from 422 patients; the independent test partition is reserved.
- **Inputs:** nodule-centred physical crops at the provider crop centre.
  - nnFoundation: z-score without clipping; reference field 64 mm, local field 32 mm.
  - FlexiCT: HU clipped to [-1000, 1000] then z-scored; reference field 72 mm at the 1.5 mm pretraining spacing, local field 50 mm.
  - nnFoundation CNN, native crop: the provider crop at native spacing without resampling (128 x 128 x 64 voxels), as recommended for nnFoundation. Its second view uses an earlier encoder stage of the same input in place of a smaller field.
- **Encoders:** loaded strictly. nnFoundation is built from the checkpoint plan as in the official classification code, FlexiCT with its official loader; outputs were checked against the official forward passes.
- **Classifiers:**
  - Reference field: embedding linear, spatial linear, embedding MLP, and mean, max and attention heads.
  - Local field: four mean+max linear descriptors and a max head.
  - Neural heads choose the learning rate on seed 2025 and repeat it with seeds 2026 and 2027.
- **Selection and comparison:** selection by development AUROC with a log-loss tie break. Intervals use 2,000 paired patient-bootstrap draws. Configuration selection used development data.

## Results

{table}

{differences}

- **FlexiCT and nnFoundation ViT:** both FlexiCT encoders and the nnFoundation ViT reach the level of the strongest existing frozen pipeline (VISTA3D) and of the official I3D; all paired AUROC differences have intervals that include zero.
- **nnFoundation CNN:** its frozen features stay far lower with every input and head tried, including the native-resolution crop.
  - The nnFoundation paper reports that the masked-autoencoding CNN can underperform when encoder features are not adapted to the task; its classification experiments use full fine-tuning.
  - With 64^3 inputs the last ResEncL stage has 2^3 positions for its instance normalization.

## Files

- `comparison.csv`: selected pipelines with patient-bootstrap intervals.
- `paired_differences.csv`: paired AUROC differences against VISTA3D and the official I3D.
- `candidates.csv`: every candidate pipeline of the new encoders.
- `provenance.json`: prediction hashes and settings.
"""


def publish(output, destination, table, provenance):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["comparison.csv", "paired_differences.csv", "candidates.csv", "comparison.png", "comparison.svg"]:
        shutil.copyfile(output / name, destination / name)
    rows = ["| Encoder | Pipeline | Seeds | AUROC [95% interval] | AP |", "|---|---|---:|---|---:|"]
    rows += [f"| {row.model} | {row.pipeline} | {row.seed_count} | {row.auroc:.4f} [{row.auroc_ci_low:.3f}, {row.auroc_ci_high:.3f}] | {row.average_precision:.4f} |" for row in table.sort_values("auroc", ascending=False).itertuples()]
    differences = pd.read_csv(output / "paired_differences.csv")
    lines = ["| Comparison | AUROC difference [95% interval] |", "|---|---|"] + [f"| {row.comparison} | {row.auroc_difference:+.4f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}] |" for row in differences.itertuples()]
    (destination / "README.md").write_text(PUBLIC_README.replace("{table}", "\n".join(rows)).replace("{differences}", "\n".join(lines)), encoding="utf-8")
    write_json(destination / "provenance.json", {**{key: value for key, value in provenance.items() if key != "predictions"},
                                                 "prediction_sha256": {name: [item["sha256"] for item in group] for name, group in provenance["predictions"].items()},
                                                 "specification": read_json(ROOT / "configs/encoder_pipeline.json"), "model_sources": {name: read_json(ROOT / "configs/model_sources.json")[name] for name in ["nnfoundation_cnn", "nnfoundation_vit", "flexict"]}})


def figure(table, output):
    colors = {"existing": "#2a78d6", "new": "#eb6834", "reference": "#52514e"}
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    order = table.sort_values("auroc").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(order) + 1.6), facecolor=surface, sharey=True)
    for axis, metric, title in zip(axes, ["auroc", "average_precision"], ["AUROC", "Average precision"]):
        for position, row in order.iterrows():
            axis.plot([row[metric + "_ci_low"], row[metric + "_ci_high"]], [position, position], color=colors[row.group], linewidth=2, solid_capstyle="round")
            axis.scatter(row[metric], position, s=64, color=colors[row.group], edgecolor=surface, linewidth=2, zorder=3)
            axis.annotate(f"{row[metric]:.3f}", (row[metric + "_ci_high"], position), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=muted)
        axis.set_xlabel(title)
        axis.set_facecolor(surface)
        axis.grid(axis="x", color=grid, linewidth=0.6)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(order)), [f"{row.model}  ({row.pipeline})" for row in order.itertuples()])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[group], label=label) for group, label in [("new", "nnFoundation / FlexiCT (frozen)"), ("existing", "existing frozen encoders"), ("reference", "official / end-to-end references")]]
    fig.suptitle("LUNA25 development: 1,285 records, 422 patients; patient-bootstrap 95% intervals", y=0.99, fontsize=10)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"comparison.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path)
    main(parser.parse_args().public)

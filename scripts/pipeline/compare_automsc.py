import argparse
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


MODELS = ["fmcib", "ctfm", "vista", "genesis", "coralbay", "tapct", "i3d", "automsc"]
LABELS = ["FMCIB", "CT-FM", "VISTA3D", "Models Genesis", "CoralBay", "TAP-CT", "Official LUNA25 I3D", "AutoMSC"]
ADDITIONAL = {"nnfoundation_cnn_native": "nnFoundation CNN", "nnfoundation_vit": "nnFoundation ViT",
              "flexict_3d": "FlexiCT-3D", "flexict_vlm": "FlexiCT-3D-VLM image encoder"}
AUTO_RUN = "20261017_automsc_mayo_development_s2025"
OUTPUT = "20261017_automsc_development"


def load_existing(include_encoders=False):
    runs = Path(context()[0]["runs"])
    selected = read_json(runs / "20260926_material_preparation/selected.json")
    paths = {name: [Path(run) / "development_predictions.csv" for run in selected[name]["repeat_runs"]] for name in MODELS[:6]}
    paths["i3d"] = [runs / "20260925_official_i3d_s2025/development_predictions.csv"]
    if include_encoders:
        suite = read_json(ROOT / "configs/encoder_pipeline.json")["suite"]
        selected = read_json(runs / suite / "selected.json")
        paths.update({name: [Path(run) / "development_predictions.csv" for run in selected[name]["repeat_runs"]] for name in ADDITIONAL})
    return paths


def tables(paths):
    result = {name: [pd.read_csv(path, dtype={"PatientID": str}).sort_values("identifier").reset_index(drop=True) for path in group] for name, group in paths.items()}
    reference = result["i3d"][0][["identifier", "PatientID", "label"]]
    assert len(reference) == 1285 and reference.PatientID.nunique() == 422
    for group in result.values():
        for table in group:
            assert table.identifier.is_unique and table.probability.between(0, 1).all()
            pd.testing.assert_frame_equal(reference, table[reference.columns])
    published = pd.read_csv(ROOT / "results/20260926_frozen_fm_development/selected.csv").set_index("model")
    if any(name in paths for name in ADDITIONAL):
        extra = pd.read_csv(ROOT / "results/20261017_encoder_pipeline/comparison.csv").set_index("key")
        published = pd.concat([published, extra.loc[list(ADDITIONAL)]])
    for name in [name for name in paths if name not in {"i3d", "automsc"}]:
        scores = []
        for table, path in zip(result[name], paths[name]):
            saved = read_json(path.parent / "result.json")
            probability = table.probability.to_numpy()
            if "seed" in saved:
                probability = probability.astype(np.float32)
            scores.append(metrics(table.label, probability))
        for metric in ["auroc", "average_precision", "brier", "log_loss"]:
            value = np.mean([score[metric] for score in scores])
            assert np.isclose(value, published.loc[name, metric], rtol=0, atol=1e-7), (name, metric, value, published.loc[name, metric])
    return result


def build(validate_existing=False, include_encoders=False):
    paths = load_existing(include_encoders)
    models = MODELS[:-1] + (list(ADDITIONAL) if include_encoders else []) + ["automsc"]
    display = LABELS[:-1] + (list(ADDITIONAL.values()) if include_encoders else []) + ["AutoMSC"]
    output_id = "20261018_all_pipeline_development" if include_encoders else OUTPUT
    runs = Path(context()[0]["runs"])
    if validate_existing:
        checked = tables(paths)
        print(f"EXISTING COMPARISON VERIFIED: {len(checked)} pipelines, 1285 identical cases, 422 patients", flush=True)
        return
    auto = runs / AUTO_RUN
    assert read_json(auto / "status.json")["status"] == "completed"
    paths["automsc"] = [auto / "predictions.csv"]
    frames = tables(paths)
    evaluation = read_json(auto / "development_evaluation.json")
    assert evaluation["prediction_sha256"] == sha256(auto / "predictions.csv")
    actual = metrics(frames["automsc"][0].label, frames["automsc"][0].probability.to_numpy())
    for metric, value in evaluation["metrics"].items():
        assert np.isclose(actual[metric], value, rtol=0, atol=1e-12), metric
    output = runs / output_id
    output.mkdir(exist_ok=False)
    labels = frames["i3d"][0].label.to_numpy()
    groups = list(frames["i3d"][0].groupby("PatientID", sort=True).indices.values())
    probabilities = {name: [frame.probability.to_numpy() for frame in frames[name]] for name in models}
    rng = np.random.default_rng(2025)
    draws = []
    for _ in tqdm(range(2000), desc="Paired patient bootstrap", mininterval=3):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(labels[indices])) < 2:
            continue
        draws.append([[np.mean([fn(labels[indices], p[indices]) for p in probabilities[name]])
                       for fn in (roc_auc_score, average_precision_score)] for name in models])
    draws = np.asarray(draws)
    assert len(draws) >= 1800
    np.save(output / "bootstrap.npy", draws)
    rows = []
    for i, (name, label) in enumerate(zip(models, display)):
        if name not in {"i3d", "automsc"}:
            scores = [read_json(path.parent / "result.json")["metrics"] for path in paths[name]]
        else:
            scores = [metrics(labels, p) for p in probabilities[name]]
        row = {"model": label, "seed_count": len(scores), "supervision": "classification and segmentation" if name == "automsc" else "classification",
               **{key: float(np.mean([score[key] for score in scores])) for key in scores[0]}}
        for j, metric in enumerate(["auroc", "average_precision"]):
            row[metric + "_ci_low"], row[metric + "_ci_high"] = np.quantile(draws[:, i, j], [.025, .975])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    deltas = []
    for i, label in enumerate(display[:-1]):
        differences = draws[:, -1, 0] - draws[:, i, 0]
        low, high = np.quantile(differences, [.025, .975])
        deltas.append({"comparison": "AutoMSC minus " + label, "auroc_difference": rows[-1]["auroc"] - rows[i]["auroc"], "ci_low": low, "ci_high": high})
    pd.DataFrame(deltas).to_csv(output / "paired_differences.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(rows) + 1.8), constrained_layout=True)
    y = np.arange(len(rows))
    colors = ["#b65d30" if name == "automsc" else "#6e7481" if name == "i3d" else "#236f88" for name in models]
    for axis, metric, title in zip(axes, ["auroc", "average_precision"], ["AUROC", "Average precision"]):
        for i, row in enumerate(rows):
            axis.plot([row[metric + "_ci_low"], row[metric + "_ci_high"]], [i, i], color=colors[i], linewidth=2)
            axis.scatter(row[metric], i, color=colors[i], s=48, zorder=3)
            axis.annotate(f"{row[metric]:.4f}", (row[metric], i), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
        axis.set(yticks=y, yticklabels=display, ylim=(len(rows) - .3, -.8), xlabel=title)
        axis.grid(axis="x", alpha=.18)
    fig.suptitle("LUNA25 development | 1,285 records, 422 patients\nPatient-bootstrap 95% intervals; existing seed aggregation retained", fontsize=12)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    provenance = {"created_at": timestamp(), "source_hash": sha256(Path(__file__)),
               "predictions": {name: [{"path": str(path), "sha256": sha256(path)} for path in group] for name, group in paths.items()},
               "resamples": len(draws), "seed": 2025, "test_used": False,
               "aggregation": "Mean of metrics across existing seeds; no averaging of prediction probabilities"}
    write_json(output / "provenance.json", provenance)
    public = ROOT / "results" / output_id
    public.mkdir(exist_ok=False)
    for name in ["comparison.csv", "paired_differences.csv", "comparison.png", "comparison.svg"]:
        (public / name).write_bytes((output / name).read_bytes())
    write_json(public / "provenance.json", {**{key: value for key, value in provenance.items() if key != "predictions"},
               "prediction_sha256": {name: [item["sha256"] for item in group] for name, group in provenance["predictions"].items()},
               "automsc_checkpoint_sha256": read_json(runs / "20261017_automsc_migration/remote/final_checkpoint_verification.json")["sha256"]})
    additional_text = ("\nThe four additional encoder configurations are fixed to the saved October 17 selection: nnFoundation CNN uses the native crop, "
                       "nnFoundation ViT uses the 32 mm field, and FlexiCT uses block 10 with 32 mm (3D) or 40 mm (VLM image encoder) fields. "
                       "The FlexiCT configurations involved additional development-based field and layer selection. "
                       "See the [encoder comparison](../20261017_encoder_pipeline/README.md) for the selection procedure.\n") if include_encoders else ""
    (public / "README.md").write_text(
        "# AutoMSC and existing LUNA25 development pipelines\n\n"
        "[Comparison figure](comparison.png) · [Metrics](comparison.csv) · [Paired AUROC differences](paired_differences.csv)\n\n"
        f"Training uses 3,729 records from 1,264 patients; development uses the same 1,285 records from 422 patients for all {len(rows)} entries. "
        "The independent test partition remains reserved. AutoMSC uses classification and segmentation supervision. "
        "The other entries use classification supervision. The six frozen encoder configurations and official I3D predictions are fixed to the existing September 26 results.\n\n"
        "AutoMSC continues the official 1,000-epoch recipe from the saved epoch-441 state on Mayo; the final checkpoint supplies the development predictions. "
        "The table preserves the existing mean of three neural-head seeds and single-fit linear results; AutoMSC and I3D each use one seed. "
        "Intervals use 2,000 paired patient bootstrap draws, retaining all records per sampled patient. Configuration selection used development data.\n" + additional_text,
        encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--include-encoders", action="store_true")
    args = parser.parse_args()
    build(args.validate_existing, args.include_encoders)

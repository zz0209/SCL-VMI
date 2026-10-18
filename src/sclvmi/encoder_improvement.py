import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .encoder_layers import SOURCES, extract, specification, suite_root
from .frozen_heads import fit_spatial


def members(directory):
    frames = [pd.read_csv(directory / f"{split}_members.csv", dtype={"PatientID": str}) for split in ["train", "development"]]
    assert set(frames[0].PatientID).isdisjoint(frames[1].PatientID)
    return frames


def bags(directory, frames, key):
    values = []
    for frame in frames:
        rows = []
        for identifier in tqdm(frame.identifier, desc=f"load {directory.parent.name} {key}", mininterval=10):
            with np.load(directory / f"{identifier}.npz") as data:
                rows.append(data[key])
        values.append(np.stack(rows))
    return values


def score(result):
    return result["metrics"]["auroc"], -result["metrics"]["log_loss"]


def candidates(name, directory, frames, inputs, root, epochs):
    spec = specification()
    return [fit_spatial(name, spec["head"], directory, frames, inputs, root, index, spec["seed"], lr, epochs=epochs) for index, lr in enumerate(spec["learning_rates"])]


def fit(name, smoke=False):
    spec = specification()
    tag = "smoke" if smoke else "features"
    root = suite_root() / ("smoke" if smoke else "heads") / name
    epochs = 2 if smoke else spec["epochs"]
    groups = {}
    for field in spec["fields"]:
        directory = Path(read_json(suite_root() / f"{name}_{field}_{tag}.json")["directory"])
        frames = members(directory)
        for block in spec["blocks"]:
            label = f"{field}_b{block:02d}"
            output = root / (directory.name if smoke else "") / label
            inputs = bags(directory, frames, f"block_{block:02d}")
            groups[label] = {"directory": directory, "frames": frames, "key": f"block_{block:02d}", "root": output, "runs": candidates(name, directory, frames, inputs, output, epochs)}
            del inputs
            gc.collect()
    baseline = read_json(Path(context()[0]["runs"]) / spec["baseline_suite"] / f"{name}_{tag}.json")["directories"]
    reference, local = Path(baseline["reference"]), Path(baseline["local"])
    if spec["two_field"] is not None:
        frames = members(local)
        for split, frame in zip(["train", "development"], members(reference)):
            assert frame.identifier.tolist() == frames[["train", "development"].index(split)].identifier.tolist()
        inputs = [np.concatenate(pair, 1) for pair in zip(bags(reference, frames, "tokens"), bags(local, frames, "tokens"))]
        output = root / (local.name if smoke else "") / "two_field_b16"
        groups["two_field_b16"] = {"directory": local, "frames": frames, "key": "two_field", "root": output, "runs": candidates(name, local, frames, inputs, output, epochs)}
        del inputs
        gc.collect()
    best = max(groups, key=lambda label: max(score(result) for result in groups[label]["runs"]))
    chosen = max(groups[best]["runs"], key=score)
    repeats = []
    if not smoke:
        group = groups[best]
        if group["key"] == "two_field":
            inputs = [np.concatenate(pair, 1) for pair in zip(bags(reference, group["frames"], "tokens"), bags(local, group["frames"], "tokens"))]
        else:
            inputs = bags(group["directory"], group["frames"], group["key"])
        repeats = [fit_spatial(name, spec["head"], group["directory"], group["frames"], inputs, group["root"], chosen["candidate"], seed, chosen["lr"], epochs=epochs) for seed in spec["repeat_seeds"]]
        del inputs
        gc.collect()
    rows = [{"model": name, "candidate": label, "lr": result["lr"], "seed": result["seed"], "run": str(group["root"] / result["run_id"]), **{metric: result["metrics"][metric] for metric in ["auroc", "average_precision", "brier", "log_loss"]}}
            for label, group in groups.items() for result in group["runs"]]
    rows += [{"model": name, "candidate": best, "lr": result["lr"], "seed": result["seed"], "run": str(groups[best]["root"] / result["run_id"]), **{metric: result["metrics"][metric] for metric in ["auroc", "average_precision", "brier", "log_loss"]}} for result in repeats]
    table = pd.DataFrame(rows)
    table.to_csv(root / "candidates.csv", index=False)
    write_json(root / "selection.json", {"model": name, "best": best, "lr": chosen["lr"], "candidate_index": chosen["candidate"], "runs": [row["run"] for row in rows if row["candidate"] == best and row["lr"] == chosen["lr"]],
                                         "completed_at": timestamp(), "smoke": smoke, "test_used": False})
    print(name, "best", best, "lr", chosen["lr"], table[table.candidate == best].auroc.round(4).tolist(), flush=True)


def report():
    spec = specification()
    runs = Path(context()[0]["runs"])
    baseline = read_json(runs / spec["baseline_suite"] / "selected.json")
    output = suite_root() / "report"
    output.mkdir(parents=True, exist_ok=True)
    tables, summary = [], {}
    for name in spec["models"]:
        table = pd.read_csv(suite_root() / "heads" / name / "candidates.csv")
        selection = read_json(suite_root() / "heads" / name / "selection.json")
        repeated = table[(table.candidate == selection["best"]) & (table.lr == selection["lr"])]
        assert len(repeated) == 1 + len(spec["repeat_seeds"])
        mean = repeated[["auroc", "average_precision", "brier", "log_loss"]].mean().to_dict()
        reference = baseline[name]["aggregate_metrics"]
        summary[name] = {"best": selection["best"], "lr": selection["lr"], "three_seed": mean, "seed_auroc": repeated.auroc.tolist(), "baseline": reference,
                         "baseline_pipeline": f"{baseline[name]['kind']} {baseline[name]['result']['arm']}", "replaces_baseline": bool(mean["auroc"] > reference["auroc"]), "runs": selection["runs"]}
        tables.append(table)
    table = pd.concat(tables)
    table.to_csv(output / "candidates.csv", index=False)
    write_json(output / "summary.json", {"models": summary, "candidates_per_model": int(table[table.seed == spec["seed"]].groupby("model").size().iloc[0]), "selection": spec["selection"],
                                         "source_sha256": {path: sha256(ROOT / path) for path in [*SOURCES, "src/sclvmi/encoder_improvement.py", "configs/encoder_improvement.json"]}, "created_at": timestamp(), "test_used": False})
    figure(table, spec, summary, output)
    for name, item in summary.items():
        print(name, item["best"], "three-seed AUROC", round(item["three_seed"]["auroc"], 4), "AP", round(item["three_seed"]["average_precision"], 4), "baseline", round(item["baseline"]["auroc"], 4), "replaces", item["replaces_baseline"], flush=True)


def figure(table, spec, summary, output):
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "text.color": ink, "xtick.color": muted, "ytick.color": muted})
    fields, blocks = list(spec["fields"]), spec["blocks"]
    fig, axes = plt.subplots(1, len(spec["models"]), figsize=(6.4 * len(spec["models"]), 3.8), facecolor=surface, squeeze=False)
    first = table[table.seed == spec["seed"]]
    grid = first[first.candidate.str.match(r"^f\d+_b\d+$")].groupby(["model", "candidate"]).auroc.max()
    low, high = np.floor(grid.min() * 100) / 100, np.ceil(grid.max() * 100) / 100
    for axis, name in zip(axes[0], spec["models"]):
        best = first[first.model == name].groupby("candidate").auroc.max()
        matrix = np.array([[best[f"{field}_b{block:02d}"] for block in blocks] for field in fields])
        image = axis.imshow(matrix, cmap="Blues", vmin=low, vmax=high)
        for row in range(len(fields)):
            for column in range(len(blocks)):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=surface if matrix[row, column] > (low + high) / 2 else ink, fontsize=9)
        axis.set_xticks(range(len(blocks)), [f"block {block}" for block in blocks])
        axis.set_yticks(range(len(fields)), [f"{spec['fields'][field]['field_mm']} mm" for field in fields])
        item = summary[name]
        two_field = f"\ntwo fields (72 + 50 mm, block 16): {best['two_field_b16']:.3f}" if "two_field_b16" in best else ""
        axis.set_title(f"{name}\nbest seed-2025 development AUROC over two learning rates, max head{two_field}\nselected {item['best']}: three-seed mean {item['three_seed']['auroc']:.3f}", fontsize=8.5, loc="left")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    for suffix in ["png", "svg"]:
        fig.savefig(output / f"layers_fields.{suffix}", dpi=170, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


def run_all(models):
    spec = specification()
    root = suite_root()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / f"attempt_{time.time_ns()}.json", {"started_at": timestamp(), "specification": spec, "models": models, "source": {path: sha256(ROOT / path) for path in [*SOURCES, "src/sclvmi/encoder_improvement.py"]}})
    for name in models:
        for smoke in [True, False]:
            for field in spec["fields"]:
                subprocess.run([sys.executable, "-u", "-m", "sclvmi.encoder_improvement", "--model", name, "--field", field, "--phase", "extract"] + (["--smoke"] if smoke else []), check=True)
            subprocess.run([sys.executable, "-u", "-m", "sclvmi.encoder_improvement", "--model", name, "--phase", "fit"] + (["--smoke"] if smoke else []), check=True)
        write_json(root / f"{name}_complete.json", {"model": name, "status": "completed", "completed_at": timestamp()})
    report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=specification()["models"])
    parser.add_argument("--field", choices=list(specification()["fields"]))
    parser.add_argument("--phase", choices=["extract", "fit", "all", "report"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "all":
        run_all(args.model)
    elif args.phase == "report":
        report()
    elif args.phase == "extract":
        extract(args.model[0], args.field, args.smoke)
    else:
        fit(args.model[0], args.smoke)


if __name__ == "__main__":
    main()

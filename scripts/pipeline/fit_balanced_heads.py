import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sclvmi.context import context, read_json, timestamp, write_json
from sclvmi.frozen_features import MODELS, SUITE


def run(name, smoke=False):
    from sclvmi.frozen_heads import fit_spatial, load_features

    storage, _, _ = context()
    runs = Path(storage["runs"])
    roots = [runs / SUITE]
    if name == "coralbay":
        roots.append(runs / (SUITE + "_coralbay_fov50"))
    candidates = [read_json(p) for root in roots for p in root.glob(f"{name}_*/result.json")]
    candidates = [r for r in candidates if r.get("seed") == 2025 and r["arm"] in ["native_mlp", "mean", "max", "attention"]]
    chosen = max(candidates, key=lambda r: (r["metrics"]["auroc"], -r["metrics"]["log_loss"]))
    directory = Path(chosen["feature_directory"])
    frames, bags, embeddings = load_features(directory, limit=16 if smoke else None)
    inputs = [x[:, None, :] for x in embeddings] if chosen["arm"] == "native_mlp" else bags
    output = runs / (SUITE + ("_balanced_smoke" if smoke else "_balanced"))
    output.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(directory / "train_members.csv")
    weight = float((membership.label == 0).sum() / (membership.label == 1).sum())
    write_json(output / f"{name}_selection.json", {"source_result": chosen, "positive_weight": weight, "decision": "Use the development-selected neural head and learning rate; compare class-balanced BCE without additional tuning."})
    for seed in ([2025] if smoke else [2025, 2026, 2027]):
        fit_spatial(name, chosen["arm"], directory, frames, inputs, output, chosen["candidate"], seed, chosen["lr"], epochs=2 if smoke else 60, positive_weight=weight)
    write_json(output / f"{name}_complete.json", {"status": "completed", "completed_at": timestamp()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run(args.models[0], smoke=args.smoke_only)
        return
    for name in args.models:
        subprocess.run([sys.executable, "-u", __file__, "--worker", "--models", name, "--smoke-only"], check=True)
        if args.smoke_only:
            continue
        subprocess.run([sys.executable, "-u", __file__, "--worker", "--models", name], check=True)
    storage, _, _ = context()
    write_json(Path(storage["runs"]) / (SUITE + ("_balanced_smoke" if args.smoke_only else "_balanced")) / "progress.json", {"status": "completed", "models": args.models, "completed_at": timestamp()})


if __name__ == "__main__":
    main()

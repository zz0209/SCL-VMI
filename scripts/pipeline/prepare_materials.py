import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.data import load_manifest, select_cases
from sclvmi.frozen_features import MODELS
from sclvmi.frozen_heads import fit_linear, fit_spatial
from sclvmi.material_features import SPEC, descriptor, experiment_root, extract, load_case, original_directory


def fit(name, smoke=False):
    root = experiment_root()
    local = Path(read_json(root / f"{name}_{'smoke' if smoke else 'features'}.json")["directory"])
    reference = original_directory(name)
    frames = [select_cases(load_manifest(), split, 8 if smoke else None) for split in ["train", "development"]]
    assert set(frames[0].PatientID).isdisjoint(frames[1].PatientID)
    arms = read_json(SPEC)["linear_arms"]
    if name == "coralbay":
        arms = [*arms, "local_native"]
    vectors = {arm: [] for arm in arms}
    bags = []
    for split, frame in zip(["train", "development"], frames):
        columns = {arm: [] for arm in arms}
        tokens = []
        for index, row in enumerate(tqdm(frame.itertuples(), total=len(frame), desc=f"{name} descriptors", mininterval=3)):
            old, new = load_case(reference, row.identifier), load_case(local, row.identifier)
            for arm in arms:
                columns[arm].append(descriptor(old, new, arm))
            tokens.append(new["tokens"])
            if index % 25 == 0 or index + 1 == len(frame):
                write_json(root / "progress.json", {"status":"running", "stage":"descriptor_loading", "model":name, "split":split, "completed":index+1, "total":len(frame), "updated_at":timestamp()})
        for arm in arms:
            vectors[arm].append(np.stack(columns[arm]))
        bags.append(np.stack(tokens))
    output = root / "smoke_heads" / name / local.name if smoke else root
    results = []
    for arm in arms:
        result = fit_linear(name, local, frames, vectors[arm], output, arm=arm)
        run = output / result["run_id"]
        write_json(run / "material_config.json", {"reference_directory": str(reference), "local_directory": str(local), "descriptor": arm, "spec_sha256": sha256(SPEC), "source_sha256": sha256(ROOT / "src/sclvmi/material_features.py"), "training_patients": int(frames[0].PatientID.nunique()), "test_used": False})
        results.append(str(run))
    del vectors
    gc.collect()
    candidates = [fit_spatial(name, "max", local, frames, bags, output / "local_heads", i, 2025, lr, epochs=2 if smoke else 60) for i, lr in enumerate([1e-4, 3e-4])]
    chosen = max(candidates, key=lambda r: (r["metrics"]["auroc"], -r["metrics"]["log_loss"]))
    if not smoke:
        for seed in [2026, 2027]:
            fit_spatial(name, "max", local, frames, bags, output / "local_heads", chosen["candidate"], seed, chosen["lr"])
    results.append(str(output / "local_heads" / chosen["run_id"]))
    write_json(output / f"{name}_candidates.json", {"model": name, "runs": results, "completed_at": timestamp()})
    print(name, [(Path(p).name, read_json(Path(p)/"result.json")["metrics"]["auroc"]) for p in results], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--phase", choices=["all", "extract", "fit"], default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = experiment_root()
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == "extract":
        extract(args.models[0], args.smoke)
        return
    if args.phase == "fit":
        fit(args.models[0], args.smoke)
        return
    stamp = timestamp()
    if (root / "experiment.json").exists():
        assert read_json(root / "experiment.json")["spec"] == read_json(SPEC), "Use a new suite for a changed experiment configuration"
    record = root / ("experiment.json" if not (root / "experiment.json").exists() else f"attempt_{stamp.replace(':', '').replace('.', '')}.json")
    write_json(record, {"started_at": stamp, "spec": read_json(SPEC), "source": {p: sha256(ROOT / p) for p in ["src/sclvmi/material_features.py", "scripts/pipeline/prepare_materials.py", "src/sclvmi/frozen_heads.py"]}, "environment_spec": sha256(ROOT / "configs/environments/pipeline.json"), "models": args.models})
    worked = False
    for name in args.models:
        completion = root / f"{name}_{'smoke_complete' if args.smoke else 'complete'}.json"
        if not args.smoke and completion.exists() and read_json(completion)["status"] == "completed":
            print("REUSE_COMPLETED", name, flush=True)
            continue
        worked = True
        for smoke in ([True] if args.smoke else [True, False]):
            for phase in ["extract", "fit"]:
                cmd = [sys.executable, "-u", __file__, "--models", name, "--phase", phase] + (["--smoke"] if smoke else [])
                subprocess.run(cmd, check=True)
        write_json(completion, {"model": name, "status": "smoke_completed" if args.smoke else "completed", "completed_at": timestamp()})
    if worked:
        write_json(root / "progress.json", {"status": "smoke_completed" if args.smoke else "completed", "models": args.models, "completed_at": timestamp()})


if __name__ == "__main__":
    main()

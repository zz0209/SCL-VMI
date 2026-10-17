import argparse
import gc
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .context import ROOT, read_json, sha256, timestamp, write_json
from .encoder_features import SOURCES, VIEWS, extract, specification, suite_root
from .frozen_heads import fit_linear, fit_spatial, load_features
from .material_features import descriptor, load_case
from .spatial_linear import spatial_vector


def fit(name, smoke=False):
    spec = specification()
    receipt = read_json(suite_root() / f"{name}_{'smoke' if smoke else 'features'}.json")
    reference, local = (Path(receipt["directories"][view]) for view in VIEWS)
    root = suite_root() / "smoke" / name / reference.name if smoke else suite_root() / "heads" / name
    root.mkdir(parents=True, exist_ok=True)
    epochs = 2 if smoke else spec["epochs"]
    frames, bags, embeddings = load_features(reference)
    runs = [root / fit_linear(name, reference, frames, embeddings, root, arm="native_linear")["run_id"], root / fit_linear(name, reference, frames, [spatial_vector(bag, name) for bag in bags], root, arm="spatial_linear")["run_id"]]
    for mode in spec["reference_heads"][2:]:
        inputs = [x[:, None, :] for x in embeddings] if mode == "native_mlp" else bags
        runs += neural(name, mode, reference, frames, inputs, root / "reference_heads", epochs, smoke)
    del bags, embeddings
    gc.collect()
    vectors, local_bags = {arm: [] for arm in spec["linear_arms"]}, []
    for frame in frames:
        columns, tokens = {arm: [] for arm in spec["linear_arms"]}, []
        for row in tqdm(frame.itertuples(), total=len(frame), desc=f"{name} local descriptors", mininterval=5):
            old, new = load_case(reference, row.identifier), load_case(local, row.identifier)
            for arm in spec["linear_arms"]:
                columns[arm].append(descriptor(old, new, arm))
            tokens.append(new["tokens"])
        for arm in spec["linear_arms"]:
            vectors[arm].append(np.stack(columns[arm]))
        local_bags.append(np.stack(tokens))
    runs += [root / fit_linear(name, local, frames, vectors[arm], root, arm=arm)["run_id"] for arm in spec["linear_arms"]]
    del vectors
    gc.collect()
    runs += neural(name, spec["local_head"], local, frames, local_bags, root / "local_heads", epochs, smoke)
    write_json(root / "candidates.json", {"model": name, "runs": [str(run) for run in runs], "completed_at": timestamp(), "smoke": smoke, "test_used": False})
    print(name, [(run.name, round(read_json(run / "result.json")["metrics"]["auroc"], 4)) for run in runs], flush=True)


def neural(name, mode, directory, frames, inputs, root, epochs, smoke):
    spec = specification()
    candidates = [fit_spatial(name, mode, directory, frames, inputs, root, index, spec["seed"], lr, epochs=epochs) for index, lr in enumerate(spec["learning_rates"])]
    selected = max(candidates, key=lambda result: (result["metrics"]["auroc"], -result["metrics"]["log_loss"]))
    repeats = [] if smoke else [fit_spatial(name, mode, directory, frames, inputs, root, selected["candidate"], seed, selected["lr"], epochs=epochs) for seed in spec["repeat_seeds"]]
    return [root / result["run_id"] for result in candidates + repeats]


def run_all(models):
    root = suite_root()
    root.mkdir(parents=True, exist_ok=True)
    sources = [*SOURCES, "src/sclvmi/encoder_pipeline.py", "src/sclvmi/frozen_heads.py", "configs/encoder_pipeline.json"]
    write_json(root / f"attempt_{time.time_ns()}.json", {"started_at": timestamp(), "specification": specification(), "models": models, "source": {path: sha256(ROOT / path) for path in sources}, "environment_spec": sha256(ROOT / "configs/environments/pipeline.json")})
    for name in models:
        for smoke in [True, False]:
            for phase in ["extract", "fit"]:
                subprocess.run([sys.executable, "-u", "-m", "sclvmi.encoder_pipeline", "--model", name, "--phase", phase] + (["--smoke"] if smoke else []), check=True)
        write_json(root / f"{name}_complete.json", {"model": name, "status": "completed", "completed_at": timestamp()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", choices=list(specification()["models"]), required=True)
    parser.add_argument("--phase", choices=["extract", "fit", "all"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.phase == "all":
        run_all(args.model)
        return
    assert len(args.model) == 1
    if args.phase == "extract":
        extract(args.model[0], args.smoke)
    else:
        fit(args.model[0], args.smoke)


if __name__ == "__main__":
    main()

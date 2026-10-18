import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .encoder_features import Fields, load


CONFIG = Path(os.environ.get("SCLVMI_ENCODER_IMPROVEMENT", ROOT / "configs/encoder_improvement.json"))
PIPELINE = ROOT / "configs/encoder_pipeline.json"
SOURCES = ["src/sclvmi/encoder_layers.py", "src/sclvmi/encoder_features.py", "src/sclvmi/models.py", "src/sclvmi/frozen_features.py", "configs/model_sources.json"]


def specification():
    return read_json(CONFIG)


def suite_root():
    storage, _, _ = context()
    return Path(storage["runs"]) / specification()["suite"]


def field_settings(name, field):
    view = specification()["fields"][field]
    model = read_json(PIPELINE)["models"][name]
    assert model["source"] == "flexict", name
    return {"shape": view["shape"], "field_mm": view["field_mm"], "spacing_mm": [view["field_mm"] / size for size in view["shape"]], "normalization": model["normalization"],
            "orientation": "LPS world, (S, P, L) array axes"}


def extract(name, field, smoke=False):
    storage, config, _ = context()
    spec = specification()
    blocks, settings = spec["blocks"], field_settings(name, field)
    request = {"model": name, "field": field, "settings": settings, "blocks": blocks, "layer_rule": spec["layer_rule"], "model_specification": read_json(PIPELINE)["models"][name],
               "dataset_revision": config["dataset_revision"], "source": {path: sha256(ROOT / path) for path in SOURCES}, "assets": read_json(ROOT / "configs/model_sources.json")["flexict"],
               "precision": "float32", "tf32": False}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(storage["activations"]) / spec["suite"] / name / field / key
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", request)
    table = load_manifest()
    frames = [select_cases(table, split, 8 if smoke else None) for split in ["train", "development"]]
    model, _ = load(name)
    model.cuda()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    root = suite_root()
    root.mkdir(parents=True, exist_ok=True)
    started, count, checks = time.perf_counter(), 0, []
    for split, frame in zip(["train", "development"], frames):
        pending = frame[[not (directory / f"{identifier}.npz").exists() for identifier in frame.identifier]]
        loader = DataLoader(Fields(pending, {field: settings}), batch_size=None, num_workers=6, persistent_workers=False, prefetch_factor=4)
        with torch.inference_mode():
            for index, (identifier, fields) in enumerate(tqdm(loader, total=len(pending), desc=f"{name} {field} {split}", mininterval=5)):
                x = fields[field][None].cuda()
                layers = model.get_intermediate_layers(x, n=[block - 1 for block in blocks], norm=True)
                values = {f"block_{block:02d}": layer[0].float().cpu().numpy() for block, layer in zip(blocks, layers)}
                assert all(np.isfinite(value).all() for value in values.values())
                if count < 4 and model.n_blocks in blocks:
                    final = model(x, is_training=True)["x_norm_patchtokens"][0].float().cpu().numpy()
                    checks.append({"identifier": identifier, "max_abs_difference": float(np.abs(final - values[f"block_{model.n_blocks:02d}"]).max())})
                    assert checks[-1]["max_abs_difference"] < 1e-4, checks[-1]
                target = directory / f"{identifier}.npz"
                with target.with_suffix(".tmp").open("wb") as handle:
                    np.savez(handle, **values, grid=np.asarray(x.shape[2:]) // model.patch_size, input_shape=np.asarray(x.shape[1:]))
                target.with_suffix(".tmp").replace(target)
                count += 1
                if index % 25 == 0 or index + 1 == len(pending):
                    write_json(root / "progress.json", {"status": "running", "stage": "features", "model": name, "field": field, "split": split, "completed": len(frame) - len(pending) + index + 1,
                                                        "total": len(frame), "new_cases": count, "seconds": time.perf_counter() - started, "updated_at": timestamp()})
        frame.to_csv(directory / f"{split}_members.csv", index=False)
    receipt = {"model": name, "field": field, "directory": str(directory), "new_cases": count, "seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
               "official_output_checks": checks, "completed_at": timestamp(), "smoke": smoke, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "test_used": False}
    write_json(root / f"{name}_{field}_{'smoke' if smoke else 'features'}.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "official_output_checks"}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return directory

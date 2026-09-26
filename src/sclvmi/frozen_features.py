import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .models import load_encoder, spatial_features
from .preprocessing import preprocess


SUITE = "20260926_frozen_heads"
MODELS = ["fmcib", "ctfm", "vista", "genesis", "coralbay", "tapct"]


def settings(name, field_mm=None):
    _, cfg, _ = context()
    if name == "coralbay":
        field = 96 if field_mm is None else field_mm
        return {"shape": [64, 64, 64], "spacing_mm": [field/64]*3, "orientation": "RAS", "intensity": [-1000, 1000], "embedding": "official_multiscale", "tokens": "last_stage", "field_mm": field}
    assert field_mm is None, "Field override currently applies to CoralBay"
    if name == "tapct":
        return {"field_mm": 50, "spacing_mm": [1, 1, 1], "orientation": "LPS", "shape": [50, 50, 50], "window": [12, 224, 224], "stride_z": 12, "embedding": "mean_window_cls", "tokens": "window_cls", "processor": "official"}
    return cfg["models"][name]


def physical_crop(path, shape, spacing, orientation):
    source = nib.load(str(path))
    center = nib.affines.apply_affine(source.affine, np.asarray(source.shape) // 2)
    signs = np.array([-1 if value in "LPI" else 1 for value in orientation])
    target_affine = np.eye(4)
    target_affine[:3, :3] = np.diag(np.asarray(spacing) * signs)
    target_affine[:3, 3] = center - target_affine[:3, :3] @ (np.asarray(shape) // 2)
    sampled = resample_from_to(source, (tuple(shape), target_affine), order=1, mode="constant", cval=-1024)
    return torch.from_numpy(np.array(sampled.dataobj, dtype=np.float32)).unsqueeze(0)


def load_fm(name):
    storage, _, _ = context()
    if name in MODELS[:4]:
        return load_encoder(name)
    os.environ["HF_MODULES_CACHE"] = str(Path(storage["project_storage"]) / "cache/hf_modules")
    os.environ["XFORMERS_DISABLED"] = "1"
    from transformers import AutoModel

    assets = read_json(Path(storage["checkpoints"]) / "published" / f"{name}.json")
    if name == "tapct":
        directory = Path(assets["directory"])
        sys.path.insert(0, str(directory.parent))
        source = importlib.import_module(f"{directory.name}.modeling_tapct")
        processor_source = importlib.import_module(f"{directory.name}.tapct_processor")
        model = source.TAPCTModel.from_pretrained(directory, local_files_only=True)
        model.processor = processor_source.TAPCTProcessor.from_pretrained(directory, local_files_only=True)
    else:
        model = AutoModel.from_pretrained(assets["directory"], trust_remote_code=True, local_files_only=True)
    return model.eval().requires_grad_(False), assets


def encode(model, name, path, config=None):
    cfg = settings(name) if config is None else config
    if name in MODELS[:4]:
        tensor = preprocess(path, name).unsqueeze(0).cuda()
        with torch.inference_mode():
            fmap = spatial_features(model, name, tensor)
            tokens = fmap.flatten(2).transpose(1, 2)[0]
            embedding = tokens.mean(0)
        return embedding.float().cpu().numpy(), tokens.float().cpu().numpy(), list(tensor.shape)
    tensor = physical_crop(path, cfg["shape"], cfg["spacing_mm"], cfg["orientation"])
    if name == "coralbay":
        tensor = ((tensor.clamp(-1000, 1000) + 1000) / 2000).unsqueeze(0).cuda()
        with torch.inference_mode():
            maps = model.encoder.forward_features(tensor)
            embedding = model.encoder.forward_encoders(maps)[0]
            tokens = maps[-1].flatten(2).transpose(1, 2)[0]
        return embedding.float().cpu().numpy(), tokens.float().cpu().numpy(), list(tensor.shape)
    tensor = model.processor(tensor.permute(0, 3, 2, 1).unsqueeze(0))["pixel_values"].cuda()
    starts = list(range(0, tensor.shape[2] - 11, 12))
    if starts[-1] != tensor.shape[2] - 12:
        starts.append(tensor.shape[2] - 12)
    features = []
    with torch.inference_mode():
        for start in starts:
            output = model(tensor[:, :, start:start + 12])
            features.append(output.pooler_output[0])
    tokens = torch.stack(features)
    return tokens.mean(0).float().cpu().numpy(), tokens.float().cpu().numpy(), list(tensor.shape)


def extract(name, limit=None, field_mm=None):
    storage, config, _ = context()
    signature = {"name": name, "settings": settings(name, field_mm), "code_sha256": sha256(Path(__file__)), "dependencies": {file: sha256(ROOT / "src/sclvmi" / file) for file in ["models.py", "preprocessing.py", "data.py"]}, "dataset_revision": config["dataset_revision"], "dataset": config["dataset"], "assets": read_json(ROOT / "configs/model_sources.json")[name], "precision": "float32", "tf32": False}
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(storage["activations"]) / SUITE / name / identity
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", signature)
    table = load_manifest()
    members = [select_cases(table, split, limit) for split in ["train", "development"]]
    model, _ = load_fm(name)
    model.cuda()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    count = 0
    progress_path = Path(storage["runs"]) / SUITE / "progress.json"
    for split, frame in zip(["train", "development"], members):
        for index, row in enumerate(tqdm(frame.itertuples(), total=len(frame), desc=f"{name} {split}", mininterval=2)):
            target = directory / f"{row.identifier}.npz"
            if not target.exists():
                embedding, tokens, shape = encode(model, name, row.image, signature["settings"])
                assert tokens.ndim == 2 and np.isfinite(tokens).all() and np.isfinite(embedding).all()
                temporary = target.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, embedding=embedding, tokens=tokens, input_shape=shape)
                temporary.replace(target)
                count += 1
            if index % 10 == 0 or index + 1 == len(frame):
                write_json(progress_path, {"status": "running", "stage": "features", "model": name, "split": split, "completed": index + 1, "total": len(frame), "new_cases": count, "seconds": time.perf_counter() - started, "updated_at": timestamp(), "directory": str(directory)})
        frame.to_csv(directory / f"{split}_members.csv", index=False)
    receipt = {"model": name, "directory": str(directory), "limit": limit, "new_cases": count, "seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "completed_at": timestamp(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    write_json(directory / f"receipt_{time.time_ns()}.json", receipt)
    key = name if field_mm is None else f"{name}_fov{field_mm:g}"
    write_json(Path(storage["runs"]) / SUITE / (f"{key}_smoke.json" if limit else f"{key}_features.json"), receipt)
    print(json.dumps(receipt), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--field-mm", type=float)
    args = parser.parse_args()
    extract(args.model, args.limit, args.field_mm)

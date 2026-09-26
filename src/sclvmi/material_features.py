import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .frozen_features import SUITE, load_fm, physical_crop, settings
from .preprocessing import preprocess


SPEC = ROOT / "configs/material_preparation.json"


def experiment_root():
    storage, _, _ = context()
    return Path(storage["runs"]) / read_json(SPEC)["suite"]


def original_directory(name):
    storage, _, _ = context()
    key = "coralbay_fov50" if name == "coralbay" else name
    return Path(read_json(Path(storage["runs"]) / SUITE / f"{key}_features.json")["directory"])


def local_settings(name):
    cfg = copy.deepcopy(settings(name, 50 if name == "coralbay" else None))
    view = read_json(SPEC)["models"][name]
    cfg.pop("crop_origin", None)
    cfg.update(shape=view["shape"], field_mm=view["field_mm"])
    cfg["crop_origin"] = "project local field specified in material_preparation.json"
    cfg["spacing_mm"] = [view["field_mm"] / n for n in view["shape"]]
    if name != "fmcib":
        cfg["spacing_origin"] = "project field_mm divided by tensor dimensions"
    return cfg


def prepare_tensor(path, name, cfg):
    if name in ["fmcib", "ctfm", "vista", "genesis"]:
        return preprocess(path, name, cfg)
    tensor = physical_crop(path, cfg["shape"], cfg["spacing_mm"], cfg["orientation"])
    return (tensor.clamp(-1000, 1000) + 1000) / 2000 if name == "coralbay" else tensor


def stage_features(model, name, tensor):
    x = tensor.unsqueeze(0).cuda()
    with torch.inference_mode():
        if name == "fmcib":
            x = model.maxpool(model.act(model.bn1(model.conv1(x))))
            values = []
            for block in [model.layer1, model.layer2, model.layer3, model.layer4]:
                x = block(x)
                values.append(x)
        elif name == "ctfm":
            values = model(x)
        elif name == "vista":
            values = model.image_encoder.encoder(x)
        elif name == "genesis":
            values = []
            for block in [model.down_tr64, model.down_tr128, model.down_tr256, model.down_tr512]:
                x, _ = block(x)
                values.append(x)
        elif name == "coralbay":
            values = model.encoder.forward_features(x)
        else:
            assert name == "tapct"
            x = model.processor(tensor.permute(0, 3, 2, 1).unsqueeze(0))["pixel_values"].cuda()
            starts = list(range(0, x.shape[2] - 11, 12))
            if starts[-1] != x.shape[2] - 12:
                starts.append(x.shape[2] - 12)
            cls, patches = [], []
            for start in starts:
                output = model(x[:, :, start:start + 12])
                cls.append(output.pooler_output[0])
                # Preserve patch channel extrema and means as window-level descriptors.
                patch = output.last_hidden_state[0]
                patches.append(torch.cat([patch.mean(0), patch.amax(0)]))
            return {"tokens": torch.stack(cls).float().cpu().numpy(), "mid": torch.stack(patches).float().cpu().numpy(), "grid": np.array([len(starts)]), "mid_grid": np.array([len(starts)]), "window_starts": np.array(starts)}
        final, middle = values[-1], values[-2]
        result = {"tokens": final.flatten(2).transpose(1, 2)[0].float().cpu().numpy(), "mid": middle.flatten(2).transpose(1, 2)[0].float().cpu().numpy(), "grid": np.array(final.shape[2:]), "mid_grid": np.array(middle.shape[2:])}
        if name == "coralbay":
            result["embedding"] = model.encoder.forward_encoders(values)[0].float().cpu().numpy()
        return result


def summarize(tokens):
    return np.concatenate([tokens.mean(0), tokens.max(0)])


def descriptor(reference, local, arm):
    if arm == "local_native":
        return local["embedding"]
    if arm == "reference_meanmax":
        return summarize(reference["tokens"])
    if arm == "local_meanmax":
        return summarize(local["tokens"])
    if arm == "local_multistage":
        return np.concatenate([summarize(local["mid"]), summarize(local["tokens"])])
    assert arm == "two_view", arm
    return np.concatenate([summarize(reference["tokens"]), summarize(local["mid"]), summarize(local["tokens"])])


def load_case(directory, identifier):
    with np.load(directory / f"{identifier}.npz") as data:
        return {key: data[key] for key in data.files}


def extract(name, smoke=False):
    storage, config, _ = context()
    root = experiment_root()
    request = {"model": name, "settings": local_settings(name), "dataset_revision": config["dataset_revision"], "source": {p: sha256(ROOT / p) for p in ["src/sclvmi/material_features.py", "src/sclvmi/preprocessing.py", "src/sclvmi/models.py", "configs/model_sources.json"]}, "spec_hash": sha256(SPEC), "precision": "float32", "tf32": False, "layers": "last and penultimate encoder stages; TAP last CLS and window patch mean/max"}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    directory = Path(storage["activations"]) / read_json(SPEC)["suite"] / name / key
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", request)
    table = load_manifest()
    frames = [select_cases(table, split, 8 if smoke else None) for split in ["train", "development"]]
    model, _ = load_fm(name)
    model.cuda()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    start, count = time.perf_counter(), 0
    for split, frame in zip(["train", "development"], frames):
        for index, row in enumerate(tqdm(frame.itertuples(), total=len(frame), desc=f"{name} local {split}", mininterval=2)):
            target = directory / f"{row.identifier}.npz"
            if not target.exists():
                tensor = prepare_tensor(row.image, name, request["settings"])
                values = stage_features(model, name, tensor)
                assert all(np.isfinite(values[k]).all() for k in ["tokens", "mid"])
                image = nib.load(row.image)
                values.update(input_shape=np.array(tensor.shape), center_ras=nib.affines.apply_affine(image.affine, np.asarray(image.shape)//2), source_affine=image.affine)
                with target.with_suffix(".tmp").open("wb") as handle:
                    np.savez(handle, **values)
                target.with_suffix(".tmp").replace(target)
                count += 1
            if index % 10 == 0 or index + 1 == len(frame):
                write_json(root / "progress.json", {"status": "running", "stage": "local_features", "model": name, "split": split, "completed": index+1, "total": len(frame), "seconds": time.perf_counter()-start, "updated_at": timestamp()})
        if not smoke:
            frame.to_csv(directory / f"{split}_members.csv", index=False)
    receipt = {"directory": str(directory), "new_cases": count, "seconds": time.perf_counter()-start, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "completed_at": timestamp(), "smoke": smoke, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    write_json(root / f"{name}_{'smoke' if smoke else 'features'}.json", receipt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return directory

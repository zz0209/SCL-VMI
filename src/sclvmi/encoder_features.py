import gc
import hashlib
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .frozen_features import physical_crop
from .models import load_flexict, load_nnfoundation


CONFIG = ROOT / "configs/encoder_pipeline.json"
VIEWS = ["reference", "local"]
SOURCES = ["src/sclvmi/encoder_features.py", "src/sclvmi/models.py", "src/sclvmi/frozen_features.py", "configs/model_sources.json"]


def specification():
    return read_json(CONFIG)


def suite_root():
    storage, _, _ = context()
    return Path(storage["runs"]) / specification()["suite"]


def view_settings(name, view):
    model = specification()["models"][name]
    settings = dict(model[view])
    if settings.get("field") == "native":
        pass
    elif view == "local":
        settings["spacing_mm"] = [settings["field_mm"] / size for size in settings["shape"]]
    else:
        settings["field_mm"] = [size * spacing for size, spacing in zip(settings["shape"], settings["spacing_mm"])]
    return {**settings, "normalization": model["normalization"], "orientation": "LPS world, (S, P, L) array axes"}


def native_crop(path):
    image = nib.load(str(path))
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt(("L", "P", "S")))
    return torch.from_numpy(np.asarray(image.as_reoriented(transform).dataobj, dtype=np.float32)).permute(2, 1, 0)


def field_tensor(path, settings):
    if settings.get("field") == "native":
        tensor = native_crop(path)
    else:
        shape, spacing = settings["shape"], settings["spacing_mm"]
        tensor = physical_crop(path, shape[::-1], spacing[::-1], "LPS")[0].permute(2, 1, 0)
        assert tuple(tensor.shape) == tuple(shape)
    if settings["normalization"] == "clip_zscore":
        tensor = tensor.clamp(-1000, 1000)
    else:
        assert settings["normalization"] == "zscore", settings["normalization"]
    tensor = (tensor - tensor.mean()) / tensor.std()
    assert torch.isfinite(tensor).all()
    return tensor[None].contiguous()


class Fields(Dataset):
    def __init__(self, frame, settings):
        self.rows, self.settings = list(zip(frame.identifier, frame.image)), settings

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        identifier, image = self.rows[index]
        return identifier, {view: field_tensor(image, settings) for view, settings in self.settings.items()}


def load(name):
    model = specification()["models"][name]
    if model["source"] == "nnfoundation_vit":
        assert model["reference"]["shape"] == model["local"]["shape"]
        return load_nnfoundation(model["source"], model["reference"]["shape"])
    if model["source"] == "nnfoundation_cnn":
        return load_nnfoundation(model["source"], None)
    return load_flexict(model["checkpoint"])


def grid(tensor):
    return np.asarray(tensor.shape[2:])


def outputs(model, name, x, view):
    spec = specification()["models"][name]
    if spec["source"] == "nnfoundation_cnn":
        stages = model(x)
        low, high = spec[view].get("stages", [-2, -1])
        final, middle = stages[high].flatten(2).transpose(1, 2)[0], stages[low].flatten(2).transpose(1, 2)[0]
        return {"embedding": final.mean(0), "tokens": final, "mid": middle, "grid": grid(stages[high]), "mid_grid": grid(stages[low])}
    if spec["source"] == "nnfoundation_vit":
        embedded = model.down_projection(x)
        tokens, rope, keep = model.eva._pos_embed(embedded.flatten(2).transpose(1, 2))
        assert keep is None and model.register_tokens is None
        for index, block in enumerate(model.eva.blocks):
            tokens = block(tokens, rope=rope)
            if index + 1 == spec["mid_block"]:
                middle = model.eva.norm(tokens)[0]
        final = model.eva.norm(tokens)[0]
        return {"embedding": final.mean(0), "tokens": final, "mid": middle, "grid": grid(embedded), "mid_grid": grid(embedded)}
    assert spec["source"] == "flexict", name
    (middle, _), (final, cls) = model.get_intermediate_layers(x, n=[spec["mid_block"] - 1, model.n_blocks - 1], return_class_token=True, norm=True)
    size = np.asarray(x.shape[2:]) // model.patch_size
    return {"embedding": cls[0], "tokens": final[0], "mid": middle[0], "grid": size, "mid_grid": size}


def check_outputs(model, name, x, values, view):
    spec = specification()["models"][name]
    if spec["source"] == "nnfoundation_vit":
        reference, _ = model.eva(model.down_projection(x).flatten(2).transpose(1, 2))
        return float((reference[0] - values["tokens"]).abs().max())
    if spec["source"] == "nnfoundation_cnn":
        return float((model(x)[spec[view].get("stages", [-2, -1])[1]].flatten(2).transpose(1, 2)[0] - values["tokens"]).abs().max())
    reference = model(x, is_training=True)
    return max(float((reference["x_norm_patchtokens"][0] - values["tokens"]).abs().max()), float((reference["x_norm_clstoken"][0] - values["embedding"]).abs().max()))


def extract(name, smoke=False):
    storage, config, _ = context()
    settings = {view: view_settings(name, view) for view in VIEWS}
    request = {"model": name, "model_specification": specification()["models"][name], "settings": settings, "dataset_revision": config["dataset_revision"],
               "source": {path: sha256(ROOT / path) for path in SOURCES}, "assets": read_json(ROOT / "configs/model_sources.json")[specification()["models"][name]["source"]], "precision": "float32", "tf32": False}
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12]
    directories = {view: Path(storage["activations"]) / specification()["suite"] / name / view / key for view in VIEWS}
    for view, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "config.json", {**request, "view": view})
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
        pending = frame[[not (directories["local"] / f"{identifier}.npz").exists() for identifier in frame.identifier]]
        loader = DataLoader(Fields(pending, settings), batch_size=None, num_workers=6, persistent_workers=False, prefetch_factor=4)
        with torch.inference_mode():
            for index, (identifier, fields) in enumerate(tqdm(loader, total=len(pending), desc=f"{name} {split}", mininterval=5)):
                for view in VIEWS:
                    x = fields[view][None].cuda()
                    values = outputs(model, name, x, view)
                    if count < 4:
                        checks.append({"identifier": identifier, "view": view, "max_abs_difference": check_outputs(model, name, x, values, view)})
                        assert checks[-1]["max_abs_difference"] < 1e-4, checks[-1]
                    values = {key: value.float().cpu().numpy() if torch.is_tensor(value) else value for key, value in values.items()}
                    assert all(np.isfinite(values[key]).all() for key in ["embedding", "tokens", "mid"])
                    target = directories[view] / f"{identifier}.npz"
                    with target.with_suffix(".tmp").open("wb") as handle:
                        np.savez(handle, **values, input_shape=np.asarray(x.shape[1:]))
                    target.with_suffix(".tmp").replace(target)
                count += 1
                if index % 25 == 0 or index + 1 == len(pending):
                    write_json(root / "progress.json", {"status": "running", "stage": "features", "model": name, "split": split, "completed": len(frame) - len(pending) + index + 1, "total": len(frame),
                                                        "new_cases": count, "seconds": time.perf_counter() - started, "updated_at": timestamp()})
        for directory in directories.values():
            frame.to_csv(directory / f"{split}_members.csv", index=False)
    receipt = {"model": name, "directories": {view: str(path) for view, path in directories.items()}, "new_cases": count, "seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
               "official_output_checks": checks, "completed_at": timestamp(), "smoke": smoke, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "test_used": False}
    write_json(root / f"{name}_{'smoke' if smoke else 'features'}.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "official_output_checks"}), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return directories

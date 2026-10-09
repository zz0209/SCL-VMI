import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import BorderPad, Compose, LoadImage, Orientation, ScaleIntensityRange, Spacing, SpatialCrop

from .context import ROOT, context, read_json
from .frozen_predict import load_head
from .models import load_encoder, spatial_features
from .sae_geometry import fmcib_sampling_affine
from .spatial_linear import spatial_vector


SITES = {
    "vista": {
        "stage2": {"module": "image_encoder.encoder.layers.2.blocks", "stride": 4, "channels": 192},
        "stage3": {"module": "image_encoder.encoder.layers.3.blocks", "stride": 8, "channels": 384},
    },
    "fmcib": {
        "layer1": {"module": "layer1", "stride": 4, "channels": 512},
        "layer2": {"module": "layer2", "stride": 8, "channels": 1024},
    },
}


def material(name):
    storage, _, _ = context()
    path = Path(storage["runs"]) / "20260926_material_preparation/materials" / name / "material.json"
    return read_json(path)


def translation(offset):
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def volume_with_geometry(path, name):
    settings = material(name)["preprocessing"]["settings"]
    source = nib.load(str(path))
    center_ras = nib.affines.apply_affine(source.affine, np.asarray(source.shape) // 2)
    if name == "fmcib":
        sys.path.insert(0, str(ROOT / "third_party/fmcib_source"))
        from fmcib.preprocessing import get_transforms

        transform = get_transforms(spatial_size=tuple(settings["shape"]))
        values = {"image_path": str(path), "coordX": -float(center_ras[0]), "coordY": -float(center_ras[1]), "coordZ": float(center_ras[2])}
        oriented = Compose(transform.transforms[:5])(values)
        image = oriented["image_path"]
        original_affine = np.asarray(image.affine).copy()
        center = nib.affines.apply_affine(np.linalg.inv(original_affine), center_ras).astype(int)
        half = np.asarray(settings["shape"])[::-1] // 2
        start = np.maximum(center - half, 0)
        end = np.minimum(center + half, image.shape[1:])
        permutation = np.eye(4)[:, [2, 1, 0, 3]]
        padding = np.maximum(np.asarray(settings["shape"]) - (end - start)[::-1], 0) // 2
        affine = original_affine @ translation(start) @ permutation @ translation(-padding)
        affine = fmcib_sampling_affine(source.affine, source.shape, affine)
        tensor = Compose(transform.transforms[5:])(oriented)
    else:
        image = LoadImage(ensure_channel_first=True, image_only=True)(str(path))
        image = Orientation(axcodes=settings["orientation"])(image)
        image = Spacing(pixdim=settings["spacing_mm"], mode="bilinear", padding_mode="border")(image)
        center = nib.affines.apply_affine(np.linalg.inv(np.asarray(image.affine)), center_ras).astype(int)
        image = ScaleIntensityRange(a_min=settings["intensity"][0], a_max=settings["intensity"][1], b_min=0, b_max=1, clip=settings["clip"])(image)
        start = center - np.asarray(settings["shape"]) // 2
        end = start + np.asarray(settings["shape"])
        before = np.maximum(-start, 0)
        after = np.maximum(end - np.asarray(image.shape[1:]), 0)
        padding = [int(value) for pair in zip(before, after) for value in pair]
        image = BorderPad(spatial_border=padding, mode="constant")(image)
        image = SpatialCrop(roi_start=(start + before).tolist(), roi_end=(end + before).tolist())(image)
        affine = np.asarray(image.affine).copy()
        tensor = image.as_tensor()
    tensor = tensor.to(dtype=torch.float32).contiguous()
    assert tuple(tensor.shape) == (1, *settings["shape"])
    assert torch.isfinite(tensor).all()
    return tensor, {"input_affine_ras": affine.tolist(), "native_affine_ras": source.affine.tolist(), "native_shape": list(source.shape), "input_shape": list(tensor.shape[1:]), "axis_order": "tensor spatial axes; flattened in C order", "origin": "index-zero convolution center", "padding_value": 0.0}


def site_affine(geometry, name, site):
    scale = np.diag([SITES[name][site]["stride"]] * 3 + [1])
    return np.asarray(geometry["input_affine_ras"]) @ scale


class SpatialEncoder:
    def __init__(self, name, device="cuda"):
        self.name = name
        self.device = device
        self.model, self.assets = load_encoder(name)
        self.model.to(device)
        self.head, self.result = load_head(Path(material(name)["run"]))
        if isinstance(self.head, torch.nn.Module):
            self.head.to(device)

    @torch.inference_mode()
    def extract(self, tensor, sites=None):
        sites = list(SITES[self.name]) if sites is None else sites
        captures = {}
        handles = []
        for site in sites:
            def capture(module, inputs, output, key=site):
                captures[key] = output.detach().clone()
            handles.append(self.model.get_submodule(SITES[self.name][site]["module"]).register_forward_hook(capture))
        final = spatial_features(self.model, self.name, tensor.unsqueeze(0).to(self.device))
        for handle in handles:
            handle.remove()
        captures["final"] = final
        return captures

    @torch.inference_mode()
    def continue_from(self, site, tensor):
        tensor = tensor.to(self.device)
        if self.name == "vista":
            index = int(site.removeprefix("stage"))
            levels = self.model.image_encoder.encoder.layers
            tensor = levels[index]["downsample"](tensor)
            for level in levels[index + 1:]:
                tensor = level["blocks"](tensor)
                tensor = level["downsample"](tensor)
        else:
            index = int(site.removeprefix("layer"))
            for number in range(index + 1, 5):
                tensor = getattr(self.model, f"layer{number}")(tensor)
        return tensor

    @torch.inference_mode()
    def probability(self, final):
        tokens = final.flatten(2).transpose(1, 2)
        if self.name == "fmcib":
            vector = spatial_vector(tokens.cpu().numpy(), self.name)
            return self.head.predict_proba(vector)[:, 1]
        return self.head(tokens).sigmoid().cpu().numpy()

    @torch.inference_mode()
    def replace(self, tensor, site, replacement):
        module = self.model.get_submodule(SITES[self.name][site]["module"])
        handle = module.register_forward_hook(lambda module, inputs, output: replacement.to(output))
        final = spatial_features(self.model, self.name, tensor.unsqueeze(0).to(self.device))
        handle.remove()
        return final

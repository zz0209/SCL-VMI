import sys

import nibabel as nib
import numpy as np
import torch
from monai.transforms import BorderPad, LoadImage, Orientation, ScaleIntensityRange, Spacing, SpatialCrop

from .context import ROOT, context


def preprocess(path, model_name):
    _, config, _ = context()
    settings = config["models"][model_name]
    source = nib.load(str(path))
    center_ras = nib.affines.apply_affine(source.affine, np.asarray(source.shape) // 2)
    if model_name == "fmcib":
        sys.path.insert(0, str(ROOT / "third_party/fmcib_source"))
        from fmcib.preprocessing import get_transforms
        transform = get_transforms(spatial_size=tuple(settings["shape"]))
        values = transform({"image_path": str(path), "coordX": -float(center_ras[0]), "coordY": -float(center_ras[1]), "coordZ": float(center_ras[2])})
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
        values = SpatialCrop(roi_start=(start + before).tolist(), roi_end=(end + before).tolist())(image).as_tensor()
    values = values.to(dtype=torch.float32).contiguous()
    assert tuple(values.shape) == (1, *settings["shape"]), values.shape
    assert torch.isfinite(values).all()
    return values

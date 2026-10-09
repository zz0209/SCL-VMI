import numpy as np
import torch
from monai.data.utils import compute_shape_offset, zoom_affine
from monai.networks.utils import normalize_transform


def fmcib_sampling_affine(native_affine, native_shape, input_affine):
    native_affine = np.asarray(native_affine, dtype=np.float64)
    input_affine = np.asarray(input_affine, dtype=np.float64)
    resampled_affine = zoom_affine(native_affine, [1.0, 1.0, 1.0], diagonal=True)
    resampled_shape, offset = compute_shape_offset(native_shape, native_affine, resampled_affine, False)
    resampled_affine[:3, 3] = offset
    if np.allclose(native_affine, resampled_affine, atol=1e-3) and np.array_equal(native_shape, resampled_shape):
        return input_affine
    source_true = normalize_transform(native_shape, align_corners=True, dtype=torch.float64).numpy()[0]
    source_false = normalize_transform(native_shape, align_corners=False, dtype=torch.float64).numpy()[0]
    target_true = normalize_transform(resampled_shape, align_corners=True, dtype=torch.float64).numpy()[0]
    target_false = normalize_transform(resampled_shape, align_corners=False, dtype=torch.float64).numpy()[0]
    # MONAI 1.5.1 AffineTransform normalizes theta with align_corners=False.
    # FMCIB samples that grid with align_corners=True; retain the actual sampling map.
    effective = native_affine @ np.linalg.inv(source_true) @ source_false @ np.linalg.inv(native_affine) @ resampled_affine @ np.linalg.inv(target_false) @ target_true
    return effective @ np.linalg.inv(resampled_affine) @ input_affine

import nibabel as nib
import numpy as np
import pytest

from sclvmi.data import load_manifest, select_cases
from sclvmi.frozen_features import physical_crop, settings


@pytest.mark.parametrize("name", ["coralbay", "tapct"])
def test_real_nodule_center_preserved(name):
    cfg = settings(name)
    rows = select_cases(load_manifest(), "train", limit=4)
    for row in rows.itertuples():
        source = nib.load(row.image)
        volume = source.get_fdata(dtype=np.float32)
        center = tuple(np.asarray(source.shape)//2)
        crop = physical_crop(row.image, cfg["shape"], cfg["spacing_mm"], cfg["orientation"])
        assert tuple(crop.shape) == (1, *cfg["shape"])
        assert np.isfinite(crop.numpy()).all()
        target_center = (0, *tuple(np.asarray(cfg["shape"])//2))
        np.testing.assert_allclose(crop[target_center].item(), volume[center], rtol=1e-5, atol=1e-3)

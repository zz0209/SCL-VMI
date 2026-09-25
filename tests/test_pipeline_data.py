import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from monai.transforms import LoadImage, Orientation, ScaleIntensityRange, Spacing

from sclvmi.context import ROOT, context
from sclvmi.data import load_manifest
from sclvmi.models import source_module
from sclvmi.preprocessing import preprocess
from sclvmi.predict import i3d_input
from sclvmi.features import preprocessed_cases


def test_patient_partition():
    table = load_manifest()
    assert table.groupby("PatientID").split.nunique().max() == 1
    assert table.identifier.is_unique
    assert table.groupby("NoduleID").split.nunique().max() == 1
    assert table.groupby("SeriesInstanceUID").split.nunique().max() == 1
    assert table.groupby("split").size().to_dict() == {"train": 3729, "development": 1285, "test": 1118}


@pytest.mark.parametrize("name", ["ctfm", "vista", "genesis"])
def test_prefetch_preserves_real_case_values_and_order(name):
    rows = list(load_manifest().query("split == 'train'").head(3).itertuples())
    results = list(preprocessed_cases(rows, name))
    assert [row.identifier for row, _ in results] == [row.identifier for row in rows]
    for row, image in results:
        np.testing.assert_array_equal(image.numpy(), preprocess(row.image, name).numpy())


def test_official_luna25_conversion():
    row = load_manifest().query("split == 'train'").iloc[0]
    source = sitk.ReadImage(row.image)
    utilities = source_module("luna25_geometry", ROOT / "third_party/luna25/preprocessing/utils.py")
    image, header = utilities.itk_image_to_numpy_image(source)
    np.testing.assert_array_equal(image, sitk.GetArrayFromImage(source))
    np.testing.assert_allclose(header["spacing"], source.GetSpacing()[::-1])
    np.testing.assert_allclose(header["origin"], source.GetOrigin()[::-1])
    assert image.shape == (64, 128, 128)


def test_i3d_inference_matches_official_dataset(tmp_path):
    from sclvmi.luna25 import configure
    directory = tmp_path
    table = load_manifest().query("split == 'development'").head(1).rename(columns={"identifier": "AnnotationID"})
    utilities = source_module("luna25_inference_test_conversion", ROOT / "third_party/luna25/preprocessing/utils.py")
    image, metadata = utilities.itk_image_to_numpy_image(sitk.ReadImage(table.iloc[0].image))
    for name, value in [("image", image), ("metadata", metadata)]:
        (directory / name).mkdir()
        np.save(directory / name / f"{table.iloc[0].AnnotationID}.npy", value)
    configure(directory, 1, 1)
    loader = source_module("luna25_dataset_equivalence", ROOT / "third_party/luna25/dataloader.py")
    dataset = loader.CTCaseDataset(directory, table, size_px=64, size_mm=50, mode="3D")
    np.testing.assert_array_equal(i3d_input(table.iloc[0].image)[0].numpy(), dataset[0]["image"].numpy())


@pytest.mark.parametrize("name", ["ctfm", "vista", "genesis"])
def test_physical_seed_preserved(name):
    row = load_manifest().query("split == 'train'").iloc[0]
    _, config, _ = context()
    settings = config["models"][name]
    volume = nib.load(row.image)
    seed = nib.affines.apply_affine(volume.affine, np.asarray(volume.shape) // 2)
    image = LoadImage(ensure_channel_first=True, image_only=True)(row.image)
    image = Orientation(axcodes=settings["orientation"])(image)
    image = Spacing(pixdim=settings["spacing_mm"], mode="bilinear", padding_mode="border")(image)
    center = nib.affines.apply_affine(np.linalg.inv(np.asarray(image.affine)), seed).astype(int)
    image = ScaleIntensityRange(a_min=settings["intensity"][0], a_max=settings["intensity"][1], b_min=0, b_max=1, clip=True)(image)
    result = preprocess(row.image, name)
    target_index = tuple(size // 2 for size in settings["shape"])
    np.testing.assert_allclose(result[(0, *target_index)], image[(0, *center)], rtol=0, atol=0)
    assert tuple(result.shape) == (1, *settings["shape"])

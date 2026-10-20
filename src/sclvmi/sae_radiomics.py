import argparse
import json
import logging
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from scipy import ndimage

from .nodule_context import contact, resample
from .nodule_mask import target_component


def as_image(array, reference):
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.CopyInformation(reference)
    return image


def record_features(task):
    row, root, config = task
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    started = time.monotonic()
    image = sitk.ReadImage(str(root / row["image"]), sitk.sitkFloat32)
    native_mask = sitk.ReadImage(str(root / row["mask"]))
    component, _ = target_component(sitk.GetArrayFromImage(native_mask))
    nodule_native = as_image(component, native_mask)
    image = resample(image, config["spacing_mm"], sitk.sitkBSpline, -1024.0)
    nodule = sitk.GetArrayFromImage(resample(nodule_native, config["spacing_mm"], sitk.sitkNearestNeighbor, 0)).astype(bool)
    assert nodule.any(), row["identifier"]
    hu = sitk.GetArrayFromImage(image)
    distance = ndimage.distance_transform_edt(~nodule)
    regions = {"nodule": nodule}
    for name, (low, high) in config["regions"].items():
        if low is not None:
            regions[name] = (distance > low) & (distance <= high)
    values = {"identifier": row["identifier"]}
    for name, classes in config["classes"].items():
        extractor = featureextractor.RadiomicsFeatureExtractor(**config["pyradiomics"])
        extractor.disableAllFeatures()
        extractor.enableImageTypes(Original={})
        for feature_class in classes:
            extractor.enableFeatureClassByName(feature_class)
        result = extractor.execute(image, as_image(regions[name], image), label=1)
        values.update({f"{name}__{key.removeprefix('original_')}": float(value) for key, value in result.items() if key.startswith("original_")})
        values[f"{name}__voxels"] = float(regions[name].sum())
    settings = config["contact"]
    measured, outside = contact(hu, nodule, settings)
    values.update({f"contact__{key}": value for key, value in measured.items()})
    for name in ["margin_0_4mm", "near_4_10mm"]:
        values[f"contact__{name}_soft_fraction"] = float((hu[regions[name]] > settings["soft_tissue_hu"]).mean())
        values[f"contact__{name}_air_fraction"] = float((hu[regions[name]] < settings["air_hu"]).mean())
        values[f"contact__{name}_outside_lung_fraction"] = float(outside[regions[name]].mean())
    values["seconds"] = time.monotonic() - started
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True, help="CSV with identifier, image and mask paths relative to --root")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    members = pd.read_csv(args.members)
    rows = members.to_dict(orient="records")[:args.limit or None]
    done = set(pd.read_csv(args.output).identifier) if args.output.exists() else set()
    pending = [row for row in rows if row["identifier"] not in done]
    print("RADIOMICS", len(pending), "records pending of", len(rows), flush=True)
    started = time.monotonic()
    buffer = []
    with multiprocessing.get_context("spawn").Pool(config["workers"]) as pool:
        for index, values in enumerate(pool.imap_unordered(record_features, [(row, args.root, config) for row in pending])):
            buffer.append(values)
            if len(buffer) == 50 or index + 1 == len(pending):
                frame = pd.DataFrame(buffer)
                frame.to_csv(args.output, mode="a", header=not args.output.exists(), index=False)
                buffer = []
                elapsed = time.monotonic() - started
                print(f"RADIOMICS {index + 1}/{len(pending)} {elapsed:.0f} s, eta {elapsed / (index + 1) * (len(pending) - index - 1):.0f} s", flush=True)
    print("RADIOMICS done", flush=True)


if __name__ == "__main__":
    main()

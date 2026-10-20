import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

from sclvmi.context import context, sha256, timestamp, write_json


RUN = "20261019_luna25_crop_check"
RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])


def official_table(luna_root):
    table = pd.read_csv(luna_root / "zenodo" / "14673658" / "LUNA25_Public_Training_Development_Data.csv")
    table["identifier"] = table.PatientID.astype(str) + "_" + table.LesionID.astype(str) + "_" + table.StudyDate.astype(str)
    assert table.identifier.is_unique
    return table


def scan_index_affine(scan):
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(scan.GetDirection()).reshape(3, 3) @ np.diag(scan.GetSpacing())
    matrix[:3, 3] = scan.GetOrigin()
    return matrix


def compare_scan(series, scan_path, cases):
    scan = sitk.ReadImage(str(scan_path))
    volume = sitk.GetArrayFromImage(scan).transpose(2, 1, 0)
    to_scan = np.linalg.inv(scan_index_affine(scan))
    rows = []
    for identifier, crop_path, coordinate in cases:
        image = nib.load(crop_path)
        assert image.header.get_slope_inter() in [(None, None), (1.0, 0.0)], identifier
        crop = np.asarray(image.dataobj)
        transform = to_scan @ RAS_TO_LPS @ image.affine
        start = np.rint(transform[:3, 3]).astype(int)
        aligned = bool(np.allclose(transform[:3, :3], np.eye(3), atol=1e-3) and np.allclose(transform[:3, 3], start, atol=1e-2))
        center = to_scan @ np.append(coordinate, 1.0)
        row = {"identifier": identifier, "SeriesInstanceUID": series, "scan_shape": list(volume.shape), "crop_shape": list(crop.shape), "aligned": aligned,
               "linear_max_deviation": float(np.abs(transform[:3, :3] - np.eye(3)).max()), "start_index": start.tolist(),
               "start_fraction_max": float(np.abs(transform[:3, 3] - start).max()), "crop_center_minus_coordinate": (start + np.asarray(crop.shape) // 2 - center[:3]).round(3).tolist()}
        if aligned:
            low, high = np.maximum(start, 0), np.minimum(start + np.asarray(crop.shape), volume.shape)
            box = tuple(slice(a - s, b - s) for a, b, s in zip(low, high, start))
            inside = np.zeros(crop.shape, dtype=bool)
            inside[box] = True
            row.update({"outside_voxels": int((~inside).sum()), "outside_values": sorted(int(value) for value in np.unique(crop[~inside]))[:5],
                        "mismatched_voxels": int((crop[box] != volume[tuple(slice(a, b) for a, b in zip(low, high))]).sum())})
        rows.append(row)
    return series, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="spread sample of this many scans; results go to a smoke directory")
    args = parser.parse_args()
    storage, _, dataset = context()
    luna_root = Path(storage["luna25_root"])
    output = Path(storage["runs"]) / (RUN + (f"_smoke{args.limit}" if args.limit else ""))
    output.mkdir(parents=True, exist_ok=True)
    official = official_table(luna_root).set_index("identifier")
    flare = pd.read_csv(dataset / "cls_data.csv", dtype={"identifier": str})
    assert set(flare.identifier) <= set(official.index)
    groups = {}
    for identifier in sorted(flare.identifier):
        row = official.loc[identifier]
        groups.setdefault(row.SeriesInstanceUID, []).append((identifier, str(dataset / "imagesTr" / f"{identifier}_0000.nii.gz"), row[["CoordX", "CoordY", "CoordZ"]].to_numpy(dtype=float)))
    series = sorted(groups)
    if args.limit:
        series = series[:: max(1, len(series) // args.limit)][: args.limit]
    partial = output / "cases.jsonl"
    done = {json.loads(line)["SeriesInstanceUID"] for line in partial.read_text(encoding="utf-8").splitlines()} if partial.exists() else set()
    todo = [name for name in series if name not in done]
    total_cases = sum(len(groups[name]) for name in series)
    print(f"crop check: {len(series)} scans, {total_cases} crops, {len(done)} scans already compared, {args.workers} workers", flush=True)
    started, finished = time.monotonic(), 0
    with ProcessPoolExecutor(args.workers) as pool, partial.open("a", encoding="utf-8") as handle:
        futures = [pool.submit(compare_scan, name, luna_root / "images" / "luna25_images" / f"{name}.mha", groups[name]) for name in todo]
        for future in as_completed(futures):
            _, rows = future.result()
            handle.write("".join(json.dumps(row) + "\n" for row in rows))
            handle.flush()
            finished += 1
            if finished % 100 == 0 or finished == len(todo):
                rate = finished / (time.monotonic() - started)
                print(f"crop check: {len(done) + finished}/{len(series)} scans, {rate:.2f} scans/s, eta {(len(todo) - finished) / rate / 60:.1f} min", flush=True)
    table = pd.DataFrame([json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()])
    table = table[table.SeriesInstanceUID.isin(series)]
    assert table.identifier.is_unique and len(table) == total_cases
    table.to_csv(output / "cases.csv", index=False)
    offsets = pd.DataFrame(table.crop_center_minus_coordinate.tolist(), columns=["x", "y", "z"])
    aligned = table[table.aligned]
    summary = {"run": RUN, "created_at": timestamp(), "scans": len(series), "crops": len(table), "aligned": len(aligned),
               "aligned_exact": int((aligned.mismatched_voxels == 0).sum()), "aligned_with_mismatch": int((aligned.mismatched_voxels > 0).sum()),
               "crops_extending_outside_scan": int((aligned.outside_voxels > 0).sum()), "outside_values": sorted({value for values in aligned.outside_values for value in values}),
               "crop_shapes": table.crop_shape.astype(str).value_counts().to_dict(), "center_offset_voxels": {axis: [float(offsets[axis].min()), float(offsets[axis].max())] for axis in offsets},
               "inputs": {"official_csv_sha256": sha256(luna_root / "zenodo" / "14673658" / "LUNA25_Public_Training_Development_Data.csv"), "flare_cls_data_sha256": sha256(dataset / "cls_data.csv")},
               "code_sha256": sha256(Path(__file__))}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

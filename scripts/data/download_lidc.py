import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = {"collection": "LIDC-IDRI", "doi": "https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX", "modality": "CT", "api": API}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def series_list(root):
    path = root / "series.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["series"]
    series = requests.get(f"{API}/getSeries", params={"Collection": COLLECTION["collection"], "Modality": COLLECTION["modality"]}, timeout=300).json()
    assert len({row["SeriesInstanceUID"] for row in series}) == len(series)
    write_json(path, {**COLLECTION, "listed_at": now(), "series": series})
    return series


def download(row, root):
    target = root / "dicom" / row["PatientID"] / row["StudyInstanceUID"] / row["SeriesInstanceUID"]
    marker = target / "verified.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8")), False
    partial = root / "partial" / f"{row['SeriesInstanceUID']}.zip"
    partial.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{API}/getImageWithMD5Hash", params={"SeriesInstanceUID": row["SeriesInstanceUID"]}, stream=True, timeout=900) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(8 * 1024 * 1024):
                handle.write(chunk)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(partial) as archive:
        hashes = {item["Filename"]: item["MD5Hash"] for item in csv.DictReader(io.TextIOWrapper(archive.open("md5hashes.csv"), encoding="utf-8"))}
        images = [name for name in archive.namelist() if name != "md5hashes.csv"]
        assert set(images) == set(hashes) and len(images) == int(row["ImageCount"]), (row["SeriesInstanceUID"], len(images), row["ImageCount"])
        total = 0
        for name in images:
            data = archive.read(name)
            assert hashlib.md5(data).hexdigest() == hashes[name], (row["SeriesInstanceUID"], name)
            (target / name).write_bytes(data)
            total += len(data)
    assert total == int(row["FileSize"]), (row["SeriesInstanceUID"], total, row["FileSize"])
    receipt = {"PatientID": row["PatientID"], "StudyInstanceUID": row["StudyInstanceUID"], "SeriesInstanceUID": row["SeriesInstanceUID"], "images": len(images), "bytes": total,
               "md5_verified": True, "archive_bytes": partial.stat().st_size, "downloaded_at": now()}
    write_json(marker, receipt)
    partial.unlink()
    return receipt, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(json.loads((ROOT / "configs/storage.local.json").read_text())["lidc_root"])
    root.mkdir(parents=True, exist_ok=True)
    series = series_list(root)
    expected = sum(int(row["FileSize"]) for row in series)
    started, done, new_bytes = time.monotonic(), 0, 0
    receipts = []
    print(f"LIDC {len(series)} CT series, {expected / 1e9:.1f} GB", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download, row, root) for row in series]
        for future in as_completed(futures):
            receipt, fresh = future.result()
            receipts.append(receipt)
            done += receipt["bytes"]
            new_bytes += receipt["bytes"] if fresh else 0
            if len(receipts) % 20 == 0 or len(receipts) == len(series):
                elapsed = time.monotonic() - started
                rate = new_bytes / max(elapsed, 1)
                print(f"LIDC {len(receipts)}/{len(series)} series, {done / 1e9:.1f}/{expected / 1e9:.1f} GB, {rate / 1e6:.1f} MB/s, eta {(expected - done) / max(rate, 1):.0f} s", flush=True)
    assert len(receipts) == len(series) and sum(item["bytes"] for item in receipts) == expected
    write_json(root / "download.json", {**COLLECTION, "license": sorted({row["LicenseName"] for row in series}), "series": len(series), "patients": len({row["PatientID"] for row in series}),
                                        "images": sum(item["images"] for item in receipts), "bytes": expected, "md5_verified": True, "completed_at": now(),
                                        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print("LIDC download verified", flush=True)


if __name__ == "__main__":
    main()

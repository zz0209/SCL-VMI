import json
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import requests
import torch

from sclvmi.context import ROOT, context, read_json, timestamp, write_json
from sclvmi.sae import SpatialDictionary


SETTINGS = read_json(ROOT / "configs/sae_interpretation.json")
CAMPAIGN = Path(context()[0]["runs"]) / SETTINGS["campaign"]
PREVIOUS = Path(context()[0]["activations"]) / SETTINGS["previous_suite"]
KEYS = {"layer1_context": ("context", "context_grid_affine_ras"), "layer2": ("layer2", "layer2_grid_affine_ras"), "stage2": ("stage2", "stage2_grid_affine_ras"), "stage3": ("stage3", "stage3_grid_affine_ras")}


def get(path, **params):
    response = requests.get("http://127.0.0.1:8773" + path, params=params, timeout=300)
    response.raise_for_status()
    return response


def independent_native(row, identifier):
    stem, affine_key = KEYS[row["site"]]
    if row["encoder"] == "random":
        path, key = CAMPAIGN / "cases" / row["model"] / f"{identifier}.npz", f"{stem}_random"
    elif row["site"] == "layer1_context":
        path, key = CAMPAIGN / "cases" / "fmcib" / f"{identifier}.npz", "context_trained"
    else:
        path, key = PREVIOUS / row["model"] / "cases" / f"{identifier}.npz", stem
    with np.load(path) as saved:
        values = saved[key]
    valid = np.ones(values.shape[1:], dtype=bool)
    if row["site"] == "layer1_context":
        with np.load(CAMPAIGN / "cases" / "fmcib" / f"{identifier}.npz") as saved:
            valid = saved["context_valid"]
    dictionary = SpatialDictionary(CAMPAIGN / "runs" / row["run_id"] / "dictionary.pt", device="cpu")
    with torch.inference_mode():
        activity = dictionary.encode(torch.from_numpy(values).flatten(1).T).numpy().T.reshape(-1, *values.shape[1:])
    return activity, valid, np.asarray(read_json(CAMPAIGN / "cases" / "fmcib" / f"{identifier}.json")[affine_key])


def check_whole(rows):
    root = Path(context()[0]["runs"]) / SETTINGS["whole_ct"]["run_id"] / "browser_cache"
    checks = []
    for row in rows:
        if "whole" not in row["scopes"] or not all((root / row["run_id"] / f"case_{case:02d}.json").exists() for case in range(8)):
            continue
        catalog = get("/api/heatmaps/features", run=row["run_id"], scope="whole").json()
        assert [item["id"] for item in catalog["features"]] == list(range(row["features"])) and all(item["ready"] for item in catalog["cases"])
        assert "lung_enrichment" in catalog["rankings"] and all("stability" in item for item in catalog["features"])
        feature = int(np.nanargmax([item["maximum"] if item["maximum"] is not None else -1 for item in catalog["features"]]))
        for case in [0, 1]:
            params = dict(run=row["run_id"], scope="whole", case=case, feature=feature, mode="spatial")
            meta = get("/api/heatmaps/map", **params).json()
            activity = np.frombuffer(get("/api/heatmaps/activity", **params).content, "<f4").reshape(meta["map_shape"])
            with h5py.File(root / row["run_id"] / f"case_{case:02d}.h5", "r") as saved:
                np.testing.assert_array_equal(activity, saved["values"][feature].astype(np.float32))
            for mode in ["single", "mean", "maximum"]:
                pooled_meta = get("/api/heatmaps/map", **{**params, "mode": mode}).json()
                pooled = np.frombuffer(get("/api/heatmaps/activity", **{**params, "mode": mode}).content, "<f4")
                assert len(pooled) == np.prod(pooled_meta["map_shape"]) and np.isfinite(pooled).all()
        checks.append({"run_id": row["run_id"], "feature": feature, "map_shape": meta["map_shape"], "scale": meta["scale"]})
        print("whole", row["run_id"], feature, meta["map_shape"], flush=True)
    return checks


def main():
    rows = [row for row in get("/api/heatmaps/catalog").json()["dictionaries"] if "nodule" in row["scopes"]]
    index = read_json(CAMPAIGN / "viewer/index.json")["dictionaries"]
    assert len(rows) == len(index) and {row["run_id"] for row in rows} == {row["run_id"] for row in index}
    members = [line.split(",")[0] for line in (CAMPAIGN / "viewer_members.csv").read_text().splitlines()[1:]]
    checks = []
    for position, row in enumerate(rows):
        started = time.monotonic()
        catalog = get("/api/heatmaps/features", run=row["run_id"], scope="nodule").json()
        assert [item["id"] for item in catalog["features"]] == list(range(row["features"])) and len(catalog["cases"]) == 256
        feature = int(np.nanargmax([item["maximum"] if item["maximum"] is not None else -1 for item in catalog["features"]]))
        case = position % 256
        identifier = members[case]
        params = dict(run=row["run_id"], scope="nodule", case=case, feature=feature, mode="native")
        meta = get("/api/heatmaps/map", **params).json()
        native_map = np.frombuffer(get("/api/heatmaps/activity", **params).content, "<f4").reshape(meta["map_shape"])
        spatial = np.frombuffer(get("/api/heatmaps/activity", **{**params, "mode": "spatial"}).content, "<f4")
        ct = np.frombuffer(get(f"/api/heatmaps/ct/nodule/{row['image_source']}/{case}").content, "<f4").reshape(meta["shape"])
        mask = np.frombuffer(get(f"/api/heatmaps/mask/nodule/{row['image_source']}/{case}").content, "<f4").reshape(meta["shape"])
        detail = get("/api/heatmaps/detail", run=row["run_id"], feature=feature).json()
        with h5py.File(CAMPAIGN / "viewer_inputs" / row["image_source"] / f"{identifier}.h5", "r") as saved:
            np.testing.assert_array_equal(ct, saved["input_hu"][...])
            np.testing.assert_array_equal(mask, saved["mask"][...].astype(np.float32))
            canonical = np.asarray(saved.attrs["canonical_affine_ras"])
        activity, valid, grid_affine = independent_native(row, identifier)
        expected = np.where(valid, activity[feature], 0)
        indices = np.argwhere(np.ones(expected.shape, dtype=bool))
        voxels = np.rint(nib.affines.apply_affine(np.linalg.inv(canonical) @ grid_affine, indices)).astype(int)
        inside = np.all((voxels >= 0) & (voxels < np.asarray(meta["shape"])), axis=1)
        observed = native_map[tuple(voxels[inside].T)]
        difference = np.abs(observed - expected[tuple(indices[inside].T)])
        assert difference.max() < 1e-4, (row["run_id"], float(difference.max()))
        assert np.isclose(meta["maximum"], expected.max(), atol=1e-4) and len(spatial) == np.prod(meta["shape"]) and np.isfinite(spatial).all()
        assert len(detail["matches"]) == len([item for item in rows if item["family"] == row["family"]]) - 1
        checks.append({"run_id": row["run_id"], "case": case, "feature": feature, "compared_positions": int(inside.sum()), "maximum_difference": float(difference.max()), "mask_voxels": int(mask.sum()), "seconds": time.monotonic() - started})
        print(f"{position + 1}/{len(rows)} {row['run_id']} feature {feature} case {case} max diff {difference.max():.2e}", flush=True)
    base = rows[0]
    for changes in [{"feature": -1}, {"feature": base["features"]}, {"case": -1}, {"case": 256}, {"mode": "unknown"}]:
        response = requests.get("http://127.0.0.1:8773/api/heatmaps/map", params={**dict(run=base["run_id"], scope="nodule", case=0, feature=0, mode="spatial"), **changes}, timeout=300)
        assert response.status_code == 422, (changes, response.status_code)
    whole_checks = check_whole(rows)
    result = {"state": "passed", "dictionaries": len(rows), "checks": checks, "whole_checks": whole_checks, "invalid_input_checks": 5, "created_at": timestamp(), "test_used": False}
    stamp = timestamp().replace(":", "").replace("+", "_")
    write_json(CAMPAIGN / "verification" / f"viewer_api_{stamp}.json", result)
    print(json.dumps({"state": "passed", "dictionaries": len(rows), "whole_dictionaries": len(whole_checks)}), flush=True)


if __name__ == "__main__":
    main()

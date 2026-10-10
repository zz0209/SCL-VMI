from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import RLock

import h5py
import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from scipy.ndimage import map_coordinates

import campaign_api
import whole_api
from sclvmi.context import ROOT, context, read_json


router = APIRouter()
READ_LOCK = RLock()
SETTINGS = read_json(ROOT / "configs/sae_whole_viewer.json")
RUN_ROOT = Path(context()[0]["runs"]) / SETTINGS["run_id"]
BROWSER_ROOT = RUN_ROOT / "browser_cache"


def dictionaries():
    selected = [{**row, "family": f"selected:{row['model']}:{row['site']}", "reference": False, "scopes": ["crop", "whole"]} for row in campaign_api.catalog_data()["dictionaries"]]
    old = read_json(Path(context()[0]["runs"]) / "20261008_spatial_sae/dictionary_index.json")
    for row in old["dictionaries"]:
        if row["default"]:
            selected.append({"run_id": row["run_id"], "model": row["model"], "site": row["site"], "seed": row["seed"], "features": row["dictionary_size"], "k": row["target_l0"], "dictionary_sha256": row["dictionary_sha256"], "family": f"reference:{row['model']}", "reference": True, "scopes": ["whole"]})
    return selected


def specification(run):
    found = [item for item in dictionaries() if item["run_id"] == run]
    if len(found) != 1:
        raise HTTPException(404, "Unknown SAE dictionary")
    return found[0]


def selected_path(run, case):
    path = BROWSER_ROOT / run / f"case_{case:02d}.h5"
    if not path.with_suffix(".json").exists():
        raise HTTPException(409, "Whole-CT browser data for this dictionary and case is still being prepared")
    return path


def browser_ready(run):
    return all((BROWSER_ROOT / run / f"case_{case:02d}.json").exists() for case in range(8))


def cases_for(spec, scope):
    if scope not in spec["scopes"]:
        raise HTTPException(422, "This dictionary has no saved data for the requested image scope")
    if scope == "crop":
        return [{"id": row["index"], "label": row["label"], "split": "development", "ready": True} for row in read_json(campaign_api.ROOT / "viewer" / spec["model"] / "cases.json")["cases"]]
    cases = whole_api.status()["cases"]
    complete = browser_ready(spec['run_id'])
    return [{"id": row["id"], "label": row["label"], "split": row["split"], "ready": row["ready"][spec["model"]] if spec["reference"] else complete} for row in cases]


def validate(run, scope, case, feature, mode):
    spec = specification(run)
    if not 0 <= feature < spec["features"]:
        raise HTTPException(422, "Feature ID is outside this dictionary")
    cases = cases_for(spec, scope)
    if not 0 <= case < len(cases):
        raise HTTPException(422, "Case is outside this image scope")
    if not cases[case]["ready"]:
        raise HTTPException(409, "This whole-CT response is still being prepared")
    modes = ["global"] if spec["site"] == "global" else ["spatial", "single", "mean", "maximum"] if scope == "whole" else ["spatial", "native"]
    if mode not in modes:
        raise HTTPException(422, "Unknown display mode for this dictionary and image scope")
    return spec


@lru_cache(maxsize=12)
def selected_map(run, case, feature, mode):
    if mode in ["single", "mean", "maximum"]:
        native, _, _ = selected_map(run, case, feature, "spatial")
    with READ_LOCK, h5py.File(selected_path(run, case), "r") as saved:
        assert saved.attrs["status"] == "completed"
        origin = np.asarray(saved.attrs["origin"])
        step = float(saved.attrs["step_mm"])
        core = int(saved.attrs["core_size"])
        if mode in ["spatial", "global"]:
            native = saved["values"][feature].astype(np.float32)
            return native, origin, step
        shape = np.asarray(native.shape)
        tiles = (shape + core - 1) // core
        lengths = [np.minimum(core, size - np.arange(count) * core) for size, count in zip(shape, tiles)]
        if mode == "single":
            middle = [np.arange(count) * core + length // 2 for count, length in zip(tiles, lengths)]
            values = native[np.ix_(*middle)]
        else:
            padded = np.pad(native, [(0, int(count * core - size)) for count, size in zip(tiles, shape)])
            blocks = padded.reshape(tiles[0], core, tiles[1], core, tiles[2], core)
            if mode == "maximum":
                values = blocks.max((1, 3, 5))
            else:
                counts = lengths[0][:, None, None] * lengths[1][None, :, None] * lengths[2][None, None, :]
                values = blocks.sum((1, 3, 5)) / counts
        return values.astype(np.float32), np.asarray(saved.attrs["tile_origin"]), float(saved.attrs["tile_step_mm"])


@lru_cache(maxsize=20)
def whole_scale(run):
    paths = [selected_path(run, case) for case in [0, 2, 4, 6]]
    maxima = []
    for path in paths:
        with READ_LOCK, h5py.File(path, "r") as saved:
            maxima.append(saved["maximum"][...])
    return np.max(maxima, axis=0)


@lru_cache(maxsize=16)
def crop_map(run, case, feature, mode):
    result = campaign_api.response_data(run, case, feature)
    if mode == "global":
        return result["native"]
    if mode == "spatial":
        return result["dense"]
    shape = campaign_api.volume_data(specification(run)["model"], case)[0].shape
    coordinates = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    native_coordinates = nib.affines.apply_affine(np.linalg.inv(result["native_to_canonical"]), coordinates)
    return map_coordinates(result["native"], native_coordinates.T, order=0, mode="nearest").reshape(shape)


@router.get("/api/heatmaps/catalog")
def catalog():
    return {"dictionaries": dictionaries(), "default_run": dictionaries()[0]["run_id"], "test_used": False}


@router.get("/api/heatmaps/features")
def features(run: str, scope: str):
    spec = specification(run)
    cases = cases_for(spec, scope)
    if spec["reference"]:
        result = whole_api.catalog(spec["model"])
        return {"features": result["features"], "cases": cases, "rankings": ["id", "lung_enrichment", "liver_enrichment", "outside_fraction", "high_hu_fraction"]}
    ready = all(row["ready"] for row in cases)
    if scope == "crop":
        with h5py.File(campaign_api.ROOT / "viewer/responses" / run / "responses.h5", "r") as saved:
            maximum = saved["maximum"][...].max(0)
    else:
        maximum = whole_scale(run) if ready else np.full(spec["features"], np.nan)
    return {"features": [{"id": index, "description": "", "maximum": float(value) if np.isfinite(value) else None} for index, value in enumerate(maximum)], "cases": cases, "rankings": ["id", "maximum"]}


@router.get("/api/heatmaps/map")
def metadata(run: str, scope: str, case: int, feature: int, mode: str):
    spec = validate(run, scope, case, feature, mode)
    if scope == "whole" and spec["reference"]:
        return {**whole_api.map_metadata(spec["model"], case, feature, mode), "pooled": False, "scope": scope, "reference": True}
    with READ_LOCK:
        if scope == "crop":
            meta = campaign_api.metadata(run, case, feature)
            values = crop_map(run, case, feature, mode)
            shape, spacing = meta["shape"], meta["spacing"]
            return {**meta, "origin": [0, 0, 0], "map_origin": [0, 0, 0], "map_shape": list(values.shape), "map_spacing_mm": spacing, "scale": max(meta["reference_maximum"], .001), "scale_source": "固定 128 名开发集患者的颜色上限", "nonzero_fraction": meta["active_fraction"], "scope": scope, "reference": False}
        values, origin, step = selected_map(run, case, feature, mode)
        volume, affine = whole_api.raw_volume(case)
        maximum = np.asarray(np.unravel_index(values.argmax(), values.shape))
        world = origin + maximum * step
        if mode != "spatial":
            world += step / 2
        peak = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), world)).astype(int)
        peak = np.clip(peak, 0, np.asarray(volume.shape) - 1)
        return {"shape": list(volume.shape), "spacing": nib.affines.voxel_sizes(affine).tolist(), "origin": affine[:3, 3].tolist(), "map_shape": list(values.shape), "map_origin": origin.tolist(), "map_spacing_mm": step, "peak": peak.tolist(), "scale": max(float(whole_scale(run)[feature]), .001), "scale_source": "固定四例 discovery 扫描的颜色上限", "maximum": float(values.max()), "mean": float(values.mean()), "nonzero_fraction": float((values > 0).mean()), "pooled": False, "scope": scope, "reference": False, "window_response": spec["site"] == "global"}


@router.get("/api/heatmaps/activity")
def activity(run: str, scope: str, case: int, feature: int, mode: str):
    spec = validate(run, scope, case, feature, mode)
    with READ_LOCK:
        if scope == "crop":
            values = crop_map(run, case, feature, mode)
        elif spec["reference"]:
            values = whole_api.activity_map(spec["model"], case, feature, mode)[0]
        else:
            values = selected_map(run, case, feature, mode)[0]
        return Response(values.astype("<f4").tobytes(), media_type="application/octet-stream")


@router.get("/api/heatmaps/ct/{scope}/{model}/{case}")
def ct(scope: str, model: str, case: int):
    if model not in ["fmcib", "vista"] or scope not in ["crop", "whole"]:
        raise HTTPException(404, "Unknown image scope or model")
    if scope == "whole":
        if not 0 <= case < 8:
            raise HTTPException(422, "Case is outside this image scope")
        return whole_api.ct_volume(case)
    return campaign_api.volume(model, case)


@router.get("/api/heatmaps/thumbnail/{scope}/{model}/{case}")
@lru_cache(maxsize=264)
def thumbnail(scope: str, model: str, case: int):
    if model not in ["fmcib", "vista"] or scope not in ["crop", "whole"]:
        raise HTTPException(404, "Unknown image scope or model")
    if scope == "whole":
        if not 0 <= case < 8:
            raise HTTPException(422, "Case is outside this image scope")
        return whole_api.thumbnail(case)
    volume, _, _ = campaign_api.volume_data(model, case)
    values = volume[:, volume.shape[1] // 2, :]
    grey = np.uint8(np.clip((values + 1000) / 1400, 0, 1).T[::-1] * 255)
    buffer = BytesIO()
    Image.fromarray(grey).resize((128, 128)).save(buffer, format="PNG")
    return Response(buffer.getvalue(), media_type="image/png")


@router.get("/api/heatmaps/preparation")
def preparation(run: str | None = None):
    result = {}
    current = specification(run) if run else None
    for model in ["fmcib", "vista"]:
        path = RUN_ROOT / f"progress_{model}.json"
        value = read_json(path) if path.exists() else {"state": "pending"}
        if value['state'] == 'completed' and not (BROWSER_ROOT / f'completed_{model}.json').exists():
            value = {**value, 'state': 'preparing_browser_cache'}
        if current and current['model'] == model:
            value = {**value, 'dictionary_ready': browser_ready(run)}
        result[model] = value
    return result

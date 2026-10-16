from functools import lru_cache
from io import BytesIO
from threading import RLock

import h5py
import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image

import interpretation_api
import whole_api


router = APIRouter()
READ_LOCK = RLock()
MODES = {"nodule": ["spatial", "native"], "whole": ["spatial", "single", "mean", "maximum"]}


def specification(run):
    found = [item for item in interpretation_api.catalog_entries() if item["run_id"] == run]
    if len(found) != 1:
        raise HTTPException(404, "Unknown SAE dictionary")
    return found[0]


def selected_path(run, case):
    path = interpretation_api.whole_path(run, case)
    if not path.with_suffix(".json").exists():
        raise HTTPException(409, "Whole-CT browser data for this dictionary and case is not available")
    return path


def cases_for(spec, scope):
    if scope not in spec["scopes"]:
        raise HTTPException(422, "This dictionary has no saved data for the requested image scope")
    if scope == "nodule":
        return interpretation_api.cases()
    return [{**row, "ready": interpretation_api.whole_ready(spec["run_id"], row["id"])} for row in whole_api.cases()]


def validate(run, scope, case, feature, mode):
    spec = specification(run)
    if not 0 <= feature < spec["features"]:
        raise HTTPException(422, "Feature ID is outside this dictionary")
    cases = cases_for(spec, scope)
    if not 0 <= case < len(cases):
        raise HTTPException(422, "Case is outside this image scope")
    if not cases[case]["ready"]:
        raise HTTPException(409, "Whole-CT browser data for this dictionary and case is not available")
    if mode not in MODES[scope]:
        raise HTTPException(422, "Unknown display mode for this image scope")
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
        if mode == "spatial":
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


@router.get("/api/heatmaps/catalog")
def catalog():
    dictionaries = interpretation_api.catalog_entries()
    return {"dictionaries": dictionaries, "default_run": dictionaries[0]["run_id"], "test_used": False}


@router.get("/api/heatmaps/features")
def features(run: str, scope: str):
    spec = specification(run)
    cases = cases_for(spec, scope)
    if scope == "nodule":
        return {"features": interpretation_api.features(run), "cases": cases, "rankings": interpretation_api.RANKINGS}
    complete = interpretation_api.whole_complete(run)
    return {"features": interpretation_api.whole_features(run), "cases": cases, "rankings": interpretation_api.WHOLE_RANKINGS if complete else ["id", "reappearance"]}


@router.get("/api/heatmaps/map")
def metadata(run: str, scope: str, case: int, feature: int, mode: str):
    validate(run, scope, case, feature, mode)
    if scope == "nodule":
        return interpretation_api.metadata(run, case, feature, mode)
    with READ_LOCK:
        values, origin, step = selected_map(run, case, feature, mode)
        volume, affine = whole_api.raw_volume(case)
        maximum = np.asarray(np.unravel_index(values.argmax(), values.shape))
        world = origin + maximum * step
        if mode != "spatial":
            world += step / 2
        peak = np.rint(nib.affines.apply_affine(np.linalg.inv(affine), world)).astype(int)
        peak = np.clip(peak, 0, np.asarray(volume.shape) - 1)
        return {"shape": list(volume.shape), "spacing": nib.affines.voxel_sizes(affine).tolist(), "origin": affine[:3, 3].tolist(), "map_shape": list(values.shape), "map_origin": origin.tolist(), "map_spacing_mm": step, "peak": peak.tolist(), "scale": max(float(whole_scale(run)[feature]), .001), "scale_source": "固定四例 discovery 扫描的颜色上限", "maximum": float(values.max()), "mean": float(values.mean()), "nonzero_fraction": float((values > 0).mean()), "scope": scope}


@router.get("/api/heatmaps/activity")
def activity(run: str, scope: str, case: int, feature: int, mode: str):
    validate(run, scope, case, feature, mode)
    with READ_LOCK:
        if scope == "nodule":
            values = interpretation_api.activity(run, case, feature, mode)[0]
        else:
            values = selected_map(run, case, feature, mode)[0]
        return Response(values.astype("<f4").tobytes(), media_type="application/octet-stream")


@router.get("/api/heatmaps/detail")
def detail(run: str, feature: int):
    specification(run)
    with READ_LOCK:
        return interpretation_api.detail(run, feature)


@router.get("/api/heatmaps/autointerp/{run_id}/{key}/{example}")
def autointerp_image(run_id: str, key: str, example: str):
    return Response(interpretation_api.autointerp_image(run_id, key, example), media_type="image/png")


@router.get("/api/heatmaps/mask/nodule/{source}/{case}")
def mask(source: str, case: int):
    with READ_LOCK:
        values = interpretation_api.volume(source, case)[1]
    return Response(values.astype("<f4").tobytes(), media_type="application/octet-stream")


@router.get("/api/heatmaps/ct/{scope}/{model}/{case}")
def ct(scope: str, model: str, case: int):
    if scope == "nodule":
        with READ_LOCK:
            values = interpretation_api.volume(model, case)[0]
        return Response(values.astype("<f4").tobytes(), media_type="application/octet-stream")
    if model not in ["fmcib", "vista"] or scope != "whole":
        raise HTTPException(404, "Unknown image scope or model")
    if not 0 <= case < len(whole_api.cases()):
        raise HTTPException(422, "Case is outside this image scope")
    return whole_api.ct_volume(case)


@router.get("/api/heatmaps/thumbnail/{scope}/{model}/{case}")
@lru_cache(maxsize=264)
def thumbnail(scope: str, model: str, case: int):
    if scope == "nodule":
        values = interpretation_api.volume(model, case)[0]
        values = values[:, values.shape[1] // 2, :]
        grey = np.uint8(np.clip((values + 1000) / 1400, 0, 1).T[::-1] * 255)
        buffer = BytesIO()
        Image.fromarray(grey).resize((128, 128)).save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")
    if model not in ["fmcib", "vista"] or scope != "whole":
        raise HTTPException(404, "Unknown image scope or model")
    if not 0 <= case < len(whole_api.cases()):
        raise HTTPException(422, "Case is outside this image scope")
    return whole_api.thumbnail(case)

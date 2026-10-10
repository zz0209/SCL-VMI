from functools import lru_cache
from pathlib import Path
from threading import RLock

import h5py
import nibabel as nib
import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse, Response
from scipy.ndimage import map_coordinates

from sclvmi.context import ROOT as PROJECT_ROOT, context, read_json
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_campaign import campaign_root, configuration


router = APIRouter()
ROOT = campaign_root()
PREVIOUS_SITES = read_json(PROJECT_ROOT / "configs/sae_preparation.json")["site_candidates"]
READ_LOCK = RLock()


def catalog_data():
    path = ROOT / "viewer/index.json"
    if not path.exists():
        return {"state": "preparing", "dictionaries": [], "test_used": False}
    return read_json(path)


def specification(run_id):
    found = [row for row in catalog_data()["dictionaries"] if row["run_id"] == run_id]
    if len(found) != 1:
        raise HTTPException(404, "Unknown qualified dictionary")
    return found[0]


def case_record(model, case):
    records = read_json(ROOT / "viewer" / model / "cases.json")["cases"]
    if not 0 <= case < len(records):
        raise HTTPException(404, "Unknown case")
    return records[case]


@lru_cache(maxsize=2)
def dictionary(run_id):
    specification(run_id)
    return SpatialDictionary(ROOT / "runs" / run_id / "dictionary.pt", device="cpu")


@lru_cache(maxsize=6)
def native_input(model, site, case):
    row = case_record(model, case)
    storage, _, _ = context()
    source = Path(storage["activations"]) / configuration()["previous_suite"] / model / "cases" if site in PREVIOUS_SITES[model] else ROOT / "cases" / model
    with np.load(source / f"{row['identifier']}.npz") as saved:
        return saved[site]


@lru_cache(maxsize=6)
def volume_data(model, case):
    case_record(model, case)
    with h5py.File(ROOT / "viewer" / model / f"case_{case:03d}.h5", "r") as saved:
        return saved["input_hu"][...], np.asarray(saved.attrs["input_affine_ras"]), np.asarray(saved.attrs["canonical_affine_ras"])


@torch.inference_mode()
def compute_response(raw, sae, feature, site_specification, volume, affine, canonical_affine):
    tokens = torch.from_numpy(raw).flatten(1).T
    if sae.request["training"].get("sae_type") == "topk":
        values = sae.encode(tokens)[:, feature]
    else:
        normalized = (tokens - sae.mean) / sae.scale - sae.weights["b_dec"]
        values = torch.relu(normalized @ sae.weights["W_enc"][:, feature])
        values *= values > sae.threshold
    native = values.numpy().reshape(raw.shape[1:])
    if site_specification["kind"] == "pooled":
        return {"native": native, "dense": None, "native_to_canonical": None, "peak": [int(size // 2) for size in volume.shape]}
    stride = site_specification["stride"]
    native_to_canonical = np.linalg.inv(canonical_affine) @ affine @ np.diag([stride] * 3 + [1])
    coordinates = np.indices(volume.shape, dtype=np.float64).reshape(3, -1).T
    native_coordinates = nib.affines.apply_affine(np.linalg.inv(native_to_canonical), coordinates)
    dense = map_coordinates(native, native_coordinates.T, order=1, mode="nearest").reshape(volume.shape)
    peak = nib.affines.apply_affine(native_to_canonical, np.asarray(np.unravel_index(native.argmax(), native.shape)))
    return {"native": native, "dense": dense, "native_to_canonical": native_to_canonical, "peak": np.rint(np.clip(peak, 0, np.asarray(volume.shape) - 1)).astype(int).tolist()}


@lru_cache(maxsize=16)
def response_data(run_id, case, feature):
    spec = specification(run_id)
    if not 0 <= feature < spec["features"]:
        raise HTTPException(404, "Feature outside dictionary range")
    raw = native_input(spec["model"], spec["site"], case)
    volume, affine, canonical_affine = volume_data(spec["model"], case)
    return compute_response(raw, dictionary(run_id), feature, spec["site_specification"], volume, affine, canonical_affine)


@router.get("/dictionaries")
def page(request: Request):
    return RedirectResponse('/whole?scope=crop&' + request.url.query)


@router.get("/api/dictionaries")
def catalog():
    value = catalog_data()
    models = {row["model"] for row in value["dictionaries"]}
    cases = {model: [{key: val for key, val in row.items() if key != "identifier"} for row in read_json(ROOT / "viewer" / model / "cases.json")["cases"]] for model in models}
    return {**value, "cases": cases}


@router.get("/api/dictionaries/{run_id}/{case}/{feature}")
def metadata(run_id: str, case: int, feature: int):
    spec = specification(run_id)
    with READ_LOCK:
        result = response_data(run_id, case, feature)
        volume, _, affine = volume_data(spec["model"], case)
    native = result["native"]
    with h5py.File(ROOT / "viewer/responses" / run_id / "responses.h5", "r") as saved:
        maxima = saved["maximum"][:, feature]
    return {"shape": list(volume.shape), "spacing": nib.affines.voxel_sizes(affine).tolist(), "native_shape": list(native.shape), "maximum": float(native.max()), "mean": float(native.mean()), "active_fraction": float((native > 0).mean()), "reference_maximum": float(maxima.max()), "strongest_case": int(maxima.argmax()), "peak": result["peak"], "pooled": spec["site"] == "global", "native_to_canonical": result["native_to_canonical"].tolist() if result["native_to_canonical"] is not None else None, "axes": ["R", "A", "S"]}


@router.get("/api/dictionaries/{run_id}/{case}/{feature}/map")
def response_map(run_id: str, case: int, feature: int):
    with READ_LOCK:
        result = response_data(run_id, case, feature)
        values = result["native"] if result["dense"] is None else result["dense"]
        return Response(values.astype("<f4").tobytes(), media_type="application/octet-stream")


@router.get("/api/dictionary-volume/{model}/{case}")
def volume(model: str, case: int):
    if model not in {row["model"] for row in catalog_data()["dictionaries"]}:
        raise HTTPException(404, "Unknown model")
    with READ_LOCK:
        values = volume_data(model, case)[0]
        return Response(values.astype("<f4").tobytes(), media_type="application/octet-stream")

from functools import lru_cache
from pathlib import Path
from threading import RLock

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from fastapi import HTTPException
from scipy.ndimage import map_coordinates

from sclvmi.context import ROOT as PROJECT_ROOT, context, read_json
from sclvmi.sae import SpatialDictionary


SETTINGS = read_json(PROJECT_ROOT / "configs/sae_interpretation.json")
ROOT = Path(context()[0]["runs"]) / SETTINGS["campaign"]
PREVIOUS = Path(context()[0]["activations"]) / SETTINGS["previous_suite"]
READ_LOCK = RLock()
SOURCES = {("fmcib", "layer1_context"): "fmcib_context", ("fmcib", "layer2"): "fmcib", ("vista", "stage2"): "vista", ("vista", "stage3"): "vista"}
ARRAYS = {
    ("fmcib", "layer1_context"): {"trained": ("new", "context_trained"), "random": ("new", "context_random"), "geometry": "context_grid_affine_ras", "valid": "context_valid"},
    ("fmcib", "layer2"): {"trained": ("previous", "layer2"), "random": ("new", "layer2_random"), "geometry": "layer2_grid_affine_ras", "valid": None},
    ("vista", "stage2"): {"trained": ("previous", "stage2"), "random": ("new", "stage2_random"), "geometry": "stage2_grid_affine_ras", "valid": None},
    ("vista", "stage3"): {"trained": ("previous", "stage3"), "random": ("new", "stage3_random"), "geometry": "stage3_grid_affine_ras", "valid": None},
}
RANKINGS = ["id", "maximum", "reappearance", "nodule_enrichment", "malignancy_partial", "image_r2", "frequency"]
WHOLE = SETTINGS["whole_ct"]
WHOLE_ROOT = Path(context()[0]["runs"]) / WHOLE["run_id"] / "browser_cache"
WHOLE_REGIONS = ["all", "lung", "liver", "other_tissue", "high_hu", "edge_air"]
WHOLE_RANKINGS = ["id", "maximum", "reappearance", "lung_enrichment", "liver_enrichment", "outside_fraction", "high_hu_fraction"]


def index():
    path = ROOT / "viewer/index.json"
    return read_json(path) if path.exists() else {"dictionaries": []}


def whole_planned(row):
    return row["seed"] in (WHOLE["trained_seeds"] if row["encoder"] == "trained" else WHOLE["random_seeds"])


def catalog_entries():
    return [{**row, "reference": False, "scopes": ["nodule", "whole"] if whole_planned(row) else ["nodule"]} for row in index()["dictionaries"]]


def whole_path(run, case):
    return WHOLE_ROOT / run / f"case_{case:02d}.h5"


def whole_ready(run, case):
    return whole_path(run, case).with_suffix(".json").exists()


def whole_complete(run):
    return all(whole_ready(run, case) for case in range(8))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


@lru_cache(maxsize=16)
def whole_statistics(run):
    assert whole_complete(run)
    sums, counts, maxima = [], [], []
    for case in range(8):
        with READ_LOCK, h5py.File(whole_path(run, case), "r") as saved:
            assert list(saved.attrs["regions"]) == WHOLE_REGIONS
            sums.append(saved["region_sums"][...])
            counts.append(saved["region_counts"][...])
            maxima.append(saved["maximum"][...])
    sums, counts = np.stack(sums)[::2], np.stack(counts)[::2]
    means = np.divide(sums, counts[:, None, :], out=np.zeros_like(sums), where=counts[:, None, :] > 0)
    active = sums[:, :, 0] > 0
    cases = active.sum(0)

    def share(region):
        values = np.divide(sums[:, :, region], sums[:, :, 0], out=np.zeros_like(sums[:, :, 0]), where=active)
        return np.divide(values.sum(0), cases, out=np.full(cases.shape, np.nan), where=cases > 0)

    return {"maximum": np.max(np.stack(maxima)[::2], axis=0), "discovery_means": means.mean(0), "outside_fraction": share(5), "high_hu_fraction": share(4)}


def whole_features(run):
    rows = features(run)
    if not whole_complete(run):
        return [{**row, "maximum": None} for row in rows]
    statistics = whole_statistics(run)
    result = []
    for row in rows:
        feature = row["id"]
        means = statistics["discovery_means"][feature]
        result.append({**row, "maximum": finite(statistics["maximum"][feature]), "lung_enrichment": ratio(means[1], means[3]), "liver_enrichment": ratio(means[2], means[3]), "outside_fraction": finite(statistics["outside_fraction"][feature]), "high_hu_fraction": finite(statistics["high_hu_fraction"][feature])})
    return result


def entry(run):
    found = [row for row in index()["dictionaries"] if row["run_id"] == run]
    if len(found) != 1:
        raise HTTPException(404, "Unknown interpretation dictionary")
    return found[0]


@lru_cache(maxsize=1)
def members():
    return pd.read_csv(ROOT / "viewer_members.csv", dtype={"PatientID": str})


@lru_cache(maxsize=1)
def references():
    return pd.read_csv(ROOT / "nodule_references.csv", dtype={"PatientID": str}).set_index("identifier")


def cases():
    frame, sizes = members(), references()
    counters = {"discovery": 0, "confirmation": 0}
    rows = []
    for index_value, row in enumerate(frame.itertuples()):
        counters[row.viewer_split] += 1
        rows.append({"id": index_value, "label": f"{'D' if row.viewer_split == 'discovery' else 'C'}{counters[row.viewer_split]:03d}", "split": row.viewer_split, "ready": True, "malignant": int(row.label), "diameter_mm": round(float(sizes.loc[row.identifier, "maximum_diameter_mm"]), 1)})
    return rows


def identifier(case):
    frame = members()
    if not 0 <= case < len(frame):
        raise HTTPException(422, "Case is outside this image scope")
    return frame.identifier.iloc[case]


@lru_cache(maxsize=48)
def volume(source, case):
    if source not in set(SOURCES.values()):
        raise HTTPException(404, "Unknown image source")
    with h5py.File(ROOT / "viewer_inputs" / source / f"{identifier(case)}.h5", "r") as saved:
        return saved["input_hu"][...].astype(np.float32), saved["mask"][...].astype(np.float32), np.asarray(saved.attrs["canonical_affine_ras"])


@lru_cache(maxsize=4)
def dictionary(run):
    entry(run)
    return SpatialDictionary(ROOT / "runs" / run / "dictionary.pt", device="cpu")


@lru_cache(maxsize=8)
def native(run, case):
    spec = entry(run)
    keys = ARRAYS[(spec["model"], spec["site"])]
    name = identifier(case)
    location, key = keys[spec["encoder"]]
    path = (ROOT / "cases" / spec["model"] / f"{name}.npz") if location == "new" else (PREVIOUS / spec["model"] / "cases" / f"{name}.npz")
    with np.load(path) as saved:
        values = saved[key]
    geometry = read_json(ROOT / "cases" / "fmcib" / f"{name}.json")
    valid = np.ones(values.shape[1:], dtype=bool)
    if keys["valid"]:
        with np.load(ROOT / "cases" / spec["model"] / f"{name}.npz") as saved:
            valid = saved[keys["valid"]]
    return values, valid, np.asarray(geometry[keys["geometry"]])


@torch.inference_mode()
def feature_native(run, case, feature):
    spec = entry(run)
    if not 0 <= feature < spec["features"]:
        raise HTTPException(422, "Feature ID is outside this dictionary")
    values, valid, affine = native(run, case)
    sae = dictionary(run)
    tokens = torch.from_numpy(values).flatten(1).T
    normalized = (tokens - sae.mean) / sae.scale - sae.weights["b_dec"]
    activity = torch.relu(normalized @ sae.weights["W_enc"][:, feature])
    activity = (activity * (activity > sae.threshold)).numpy().reshape(values.shape[1:])
    return np.where(valid, activity, 0).astype(np.float32), valid, affine


@lru_cache(maxsize=16)
def activity(run, case, feature, mode):
    spec = entry(run)
    native_values, valid, affine = feature_native(run, case, feature)
    hu, _, canonical_affine = volume(SOURCES[(spec["model"], spec["site"])], case)
    native_to_canonical = np.linalg.inv(canonical_affine) @ affine
    coordinates = np.indices(hu.shape, dtype=np.float64).reshape(3, -1).T
    position = nib.affines.apply_affine(np.linalg.inv(native_to_canonical), coordinates)
    context_grid = spec["site"] == "layer1_context"
    if mode == "spatial":
        values = map_coordinates(native_values, position.T, order=1, mode="constant" if context_grid else "nearest", cval=0.0)
    else:
        rounded = np.rint(position).astype(int)
        inside = np.all((rounded >= 0) & (rounded < np.asarray(native_values.shape)), axis=1)
        clipped = np.clip(rounded, 0, np.asarray(native_values.shape) - 1)
        values = native_values[tuple(clipped.T)]
        if context_grid:
            values = np.where(inside, values, 0)
    return values.reshape(hu.shape).astype(np.float32), native_to_canonical


@lru_cache(maxsize=8)
def profile_arrays(model, site):
    directory = ROOT / "profiles" / f"{model}_{site}"
    with np.load(directory / "profiles.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    return arrays, read_json(directory / "summary.json")


@lru_cache(maxsize=8)
def stability_arrays(group):
    directory = ROOT / "stability" / group
    with np.load(directory / "features.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    return arrays, read_json(directory / "summary.json")


def percentile(values, reference):
    clean = np.sort(reference[np.isfinite(reference)])
    result = np.searchsorted(clean, values, side="right") / max(len(clean), 1) * 100
    return np.where(np.isfinite(values), result, np.nan)


@lru_cache(maxsize=16)
def feature_table(run):
    spec = entry(run)
    arrays, summary = profile_arrays(spec["model"], spec["site"])
    prefix = f"{spec['encoder']}_{spec['seed']}_"
    random_prefix = f"random_{SETTINGS['grid']['reference_seed']}_"
    stability, _ = stability_arrays(spec["stability_group"])
    seeds = list(stability["seeds"])
    position = seeds.index(spec["seed"])
    reappearance = stability["reappearance"][position].astype(int)
    live = stability["development_live"][position]
    maxima = arrays[prefix + "viewer_maximum"]
    table = {
        "maximum": maxima.max(0), "strongest": maxima.argmax(0), "reappearance": reappearance, "live": live,
        "nodule_enrichment": arrays[prefix + "nodule_enrichment"], "frequency": arrays[prefix + "frequency"],
        "partial_train": arrays[prefix + "partial_train"], "partial_development": arrays[prefix + "partial_development"],
        "development_auroc": arrays[prefix + "development_auroc"], "region_mean": arrays[prefix + "region_mean"], "overall_mean": arrays[prefix + "overall_mean"],
        "image_r2": arrays.get(prefix + "image_r2"),
        "enrichment_percentile": percentile(arrays[prefix + "nodule_enrichment"], arrays[random_prefix + "nodule_enrichment"]),
        "partial_percentile": percentile(np.abs(arrays[prefix + "partial_development"]), np.abs(arrays[random_prefix + "partial_development"])),
    }
    return table, summary, stability, position


def stability_label(count, live, total):
    if not live:
        return "inactive"
    if count >= SETTINGS["stability"]["stable_minimum_reappearance"]:
        return "stable"
    if count <= SETTINGS["stability"]["unstable_maximum_reappearance"]:
        return "unstable"
    return "partial"


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def features(run):
    table, _, stability, _ = feature_table(run)
    total = len(stability["seeds"]) - 1
    rows = []
    for feature in range(len(table["maximum"])):
        rows.append({
            "id": feature, "description": "", "maximum": finite(table["maximum"][feature]),
            "reappearance": int(table["reappearance"][feature]), "stability": stability_label(table["reappearance"][feature], table["live"][feature], total),
            "nodule_enrichment": finite(table["nodule_enrichment"][feature]), "malignancy_partial": finite(abs(table["partial_train"][feature])) if np.isfinite(table["partial_train"][feature]) else None,
            "image_r2": finite(table["image_r2"][feature]) if table["image_r2"] is not None else None, "frequency": finite(table["frequency"][feature]),
        })
    return rows


def detail(run, feature):
    spec = entry(run)
    table, summary, stability, position = feature_table(run)
    if not 0 <= feature < len(table["maximum"]):
        raise HTTPException(422, "Feature ID is outside this dictionary")
    seeds = list(stability["seeds"])
    group = [row for row in index()["dictionaries"] if row["family"] == spec["family"]]
    by_seed = {row["seed"]: row["run_id"] for row in group}
    matches = []
    for target, seed in enumerate(seeds):
        if target == position:
            continue
        matches.append({"seed": int(seed), "run_id": by_seed.get(int(seed)), "feature": int(stability["matched"][position, target, feature]), "decoder_cosine": finite(stability["best_cosine"][position, target, feature]), "activation_correlation": finite(stability["matched_correlation"][position, target, feature]), "best_activation_correlation": finite(stability["best_correlation"][position, target, feature])})
    region = {name: finite(table["region_mean"][index_value, feature]) for index_value, name in enumerate(summary["bins"])}
    return {
        "run_id": run, "feature": feature, "encoder": spec["encoder"], "seed": spec["seed"],
        "stability": stability_label(table["reappearance"][feature], table["live"][feature], len(seeds) - 1), "reappearance": int(table["reappearance"][feature]), "comparisons": len(seeds) - 1,
        "cosine_threshold": SETTINGS["stability"]["decoder_cosine"], "matches": matches,
        "region_mean": region, "overall_mean": finite(table["overall_mean"][feature]), "nodule_enrichment": finite(table["nodule_enrichment"][feature]),
        "frequency": finite(table["frequency"][feature]), "partial_train": finite(table["partial_train"][feature]), "partial_development": finite(table["partial_development"][feature]),
        "development_auroc": finite(table["development_auroc"][feature]), "image_r2": finite(table["image_r2"][feature]) if table["image_r2"] is not None else None,
        "enrichment_percentile": finite(table["enrichment_percentile"][feature]), "partial_percentile": finite(table["partial_percentile"][feature]),
        "size_only_development_auroc": summary["size_only_development_auroc"], "summary_rule": summary["summary_rule"],
        "discovery_patients": summary["discovery_patients"], "confirmation_patients": summary["confirmation_patients"],
        "strongest_case": int(table["strongest"][feature]),
    }


def metadata(run, case, feature, mode):
    spec = entry(run)
    with READ_LOCK:
        values, native_to_canonical = activity(run, case, feature, mode)
        native_values, valid, _ = feature_native(run, case, feature)
        hu, _, canonical_affine = volume(SOURCES[(spec["model"], spec["site"])], case)
    table, _, _, _ = feature_table(run)
    peak_native = np.asarray(np.unravel_index(native_values.argmax(), native_values.shape))
    peak = np.rint(np.clip(nib.affines.apply_affine(native_to_canonical, peak_native), 0, np.asarray(hu.shape) - 1)).astype(int)
    spacing = nib.affines.voxel_sizes(canonical_affine).tolist()
    return {
        "shape": list(hu.shape), "spacing": spacing, "origin": [0, 0, 0], "map_origin": [0, 0, 0], "map_shape": list(values.shape), "map_spacing_mm": spacing,
        "native_shape": list(native_values.shape), "maximum": float(native_values.max()), "mean": float(native_values[valid].mean()), "nonzero_fraction": float((native_values[valid] > 0).mean()),
        "valid_positions": int(valid.sum()), "native_positions": int(valid.size),
        "scale": max(float(table["maximum"][feature]), .001), "scale_source": "固定 256 名浏览患者的颜色上限", "strongest_case": int(table["strongest"][feature]),
        "peak": peak.tolist(), "pooled": False, "scope": "nodule", "reference": False, "mask": True,
    }

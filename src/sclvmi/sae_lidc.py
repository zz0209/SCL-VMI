import argparse
import hashlib
import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import pylidc as pl
import SimpleITK as sitk
import torch
from pylidc.utils import consensus
from scipy import ndimage, stats

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .models import load_encoder
from .nodule_context import contact, resample
from .nodule_mask import target_component
from .sae import SpatialDictionary
from .sae_concepts import dictionaries, spearman_matrix, stable_mask
from .sae_context import RecordDataset, configuration as campaign_configuration, unchanged, vista_sites
from .sae_encoders import fmcib_prefix, parameter_digest, random_encoder
from .sae_interp_candidates import correlate, residual_ranks


CONFIG_PATH = ROOT / "configs/sae_lidc.json"
RADIOMICS_CONFIG = ROOT / "configs/sae_radiomics.json"
RAS_FROM_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
SITE_ARRAYS = {"fmcib": ("layer2", "layer2_random", "layer2_distance"), "vista": ("stage2", "stage2_random", "stage2_distance")}
CONTACT = ["lung_boundary_contact_fraction", "lung_boundary_distance_mm", "margin_0_4mm_soft_fraction", "near_4_10mm_soft_fraction", "margin_0_4mm_outside_lung_fraction", "near_4_10mm_outside_lung_fraction"]


def configuration():
    return read_json(CONFIG_PATH)


def directories():
    config, storage = configuration(), context()[0]
    return Path(storage["lidc_root"]), Path(storage["preprocessed"]) / "lidc" / config["run_id"], Path(storage["runs"]) / config["run_id"], Path(storage["runs"]) / campaign_configuration()["campaign"]


def translation(offset):
    matrix = np.eye(4)
    matrix[:3, 3] = offset
    return matrix


def scan_volume(scan, lidc_root, settings):
    directory = lidc_root / "dicom" / scan.patient_id / scan.study_instance_uid / scan.series_instance_uid
    images = [pydicom.dcmread(path) for path in sorted(directory.glob("*.dcm"))]
    images = [image for image in images if str(image.SeriesInstanceUID).strip() == scan.series_instance_uid and str(image.StudyInstanceUID).strip() == scan.study_instance_uid]
    # pylidc rule: a repeated slice position keeps the image with the lower InstanceNumber; slices are sorted by position
    chosen = {}
    for image in sorted(images, key=lambda image: float(image.InstanceNumber)):
        chosen.setdefault(float(image.ImagePositionPatient[2]), image)
    positions = np.asarray(sorted(chosen))
    images = [chosen[position] for position in positions]
    reasons = []
    database = scan.slice_zvals
    if len(positions) != len(database) or np.abs(positions - database).max() > settings["slice_spacing_tolerance"]:
        reasons.append("slice positions differ from the pylidc database")
    gaps = np.diff(positions)
    if np.abs(gaps / np.median(gaps) - 1).max() > settings["slice_spacing_tolerance"]:
        reasons.append("irregular slice gaps")
    orientation = np.asarray([[float(value) for value in image.ImageOrientationPatient] for image in images])
    if not np.allclose(orientation, [1, 0, 0, 0, 1, 0], atol=1e-3):
        reasons.append("orientation is not axial")
    origins = np.asarray([[float(value) for value in image.ImagePositionPatient[:2]] for image in images])
    spacings = np.asarray([[float(value) for value in image.PixelSpacing] for image in images])
    if not (np.allclose(origins, origins[0], atol=1e-3) and np.allclose(spacings, spacings[0], atol=1e-6)):
        reasons.append("in-plane origin or pixel spacing varies")
    if reasons:
        return None, None, reasons
    # pylidc to_volume: (rows, columns, slices) = annotation index (i, j, k)
    volume = np.stack([image.pixel_array * float(image.RescaleSlope) + float(image.RescaleIntercept) for image in images], axis=-1).astype(np.int16)
    row_spacing, column_spacing = spacings[0]
    affine = np.eye(4)
    affine[:3, :3] = np.diag([column_spacing, row_spacing, float(np.median(gaps))])
    affine[:3, 3] = [origins[0, 0], origins[0, 1], positions[0]]
    return volume.transpose(1, 0, 2), affine, []


def crop(volume, start, shape, fill):
    output = np.full(shape, fill, dtype=volume.dtype)
    low, high = np.maximum(start, 0), np.minimum(start + np.asarray(shape), volume.shape)
    if np.all(high > low):
        output[tuple(slice(a - s, b - s) for a, b, s in zip(low, high, start))] = volume[tuple(slice(a, b) for a, b in zip(low, high))]
    return output


def save_nifti(values, affine_lps, path):
    image = nib.Nifti1Image(values, RAS_FROM_LPS @ affine_lps)
    image.set_qform(RAS_FROM_LPS @ affine_lps, code=1)
    image.set_sform(RAS_FROM_LPS @ affine_lps, code=1)
    nib.save(image, path)


def contact_measures(hu, mask, affine_lps, settings):
    def as_image(values, dtype):
        image = sitk.GetImageFromArray(np.ascontiguousarray(values.transpose(2, 1, 0)).astype(dtype))
        image.SetSpacing(tuple(float(value) for value in np.diag(affine_lps)[:3]))
        image.SetOrigin(tuple(float(value) for value in affine_lps[:3, 3]))
        return image

    component, _ = target_component(mask)
    image = resample(as_image(hu, np.float32), settings["spacing_mm"], sitk.sitkBSpline, -1024.0)
    nodule = sitk.GetArrayFromImage(resample(as_image(component, np.uint8), settings["spacing_mm"], sitk.sitkNearestNeighbor, 0)).astype(bool)
    assert nodule.any()
    values = sitk.GetArrayFromImage(image)
    measured, outside = contact(values, nodule, settings)
    distance = ndimage.distance_transform_edt(~nodule)
    for name, (low, high) in {"margin_0_4mm": (0.0, 4.0), "near_4_10mm": (4.0, 10.0)}.items():
        region = (distance > low) & (distance <= high)
        measured[f"{name}_soft_fraction"] = float((values[region] > settings["soft_tissue_hu"]).mean())
        measured[f"{name}_outside_lung_fraction"] = float(outside[region].mean())
    return measured


def process_scan(scan_id, lidc_root, output):
    config = configuration()
    settings, shape = config["nodules"], np.asarray(config["crop"]["shape"])
    contact_settings = {**read_json(RADIOMICS_CONFIG)["contact"], "spacing_mm": config["contact"]["spacing_mm"]}
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    receipt = {"scan_id": scan_id, "PatientID": scan.patient_id, "SeriesInstanceUID": scan.series_instance_uid, "nodules": [], "readers": [], "exclusions": []}
    if scan_id in settings["cluster_failures"]:
        receipt["clusters_with_enough_readers"] = 0
        receipt["exclusions"].append({"scan_id": scan_id, "cluster": None, "reason": "annotations cannot be clustered by pylidc"})
        return receipt
    clusters =[(index, cluster) for index, cluster in enumerate(scan.cluster_annotations(verbose=False)) if settings["minimum_readers"] <= len(cluster) <= settings["maximum_readers"]]
    receipt["clusters_with_enough_readers"] = len(clusters)
    if not clusters:
        return receipt
    volume, affine, reasons = scan_volume(scan, lidc_root, settings)
    if reasons:
        receipt["exclusions"].append({"scan_id": scan_id, "cluster": None, "reason": "; ".join(reasons)})
        return receipt
    voxel_volume = float(np.prod(np.diag(affine)[:3]))
    for index, cluster in clusters:
        identifier = f"{scan.patient_id}_s{scan_id}_n{index}"
        mask, bbox, _ = consensus(cluster, clevel=settings["consensus_level"], ret_masks=True)
        if not mask.any():
            receipt["exclusions"].append({"scan_id": scan_id, "cluster": index, "reason": "empty consensus mask"})
            continue
        points = np.argwhere(mask) + np.asarray([value.start for value in bbox])
        points = points[:, [1, 0, 2]]
        center = np.rint(points.mean(0)).astype(int)
        start = center - shape // 2
        full = np.zeros(volume.shape, dtype=np.uint8)
        full[points[:, 0], points[:, 1], points[:, 2]] = 1
        image, nodule = crop(volume, start, shape, config["crop"]["pad_hu"]), crop(full, start, shape, 0)
        crop_affine = affine @ translation(start)
        paths = {"image": output / "crops" / f"{identifier}.nii.gz", "mask": output / "masks" / f"{identifier}.nii.gz"}
        save_nifti(image, crop_affine, paths["image"])
        save_nifti(nodule, crop_affine, paths["mask"])
        ratings = [{"identifier": identifier, "annotation_id": annotation.id, **{name: int(getattr(annotation, name)) for name in ["subtlety", "internalStructure", "calcification", "sphericity", "margin", "lobulation", "spiculation", "texture", "malignancy"]}} for annotation in cluster]
        receipt["readers"].extend(ratings)
        frame = pd.DataFrame(ratings)
        center_lps = affine @ np.append(points.mean(0), 1.0)
        receipt["nodules"].append({
            "identifier": identifier, "PatientID": scan.patient_id, "scan_id": scan_id, "cluster": index, "SeriesInstanceUID": scan.series_instance_uid, "StudyInstanceUID": scan.study_instance_uid,
            "readers": len(cluster), "consensus_voxels": int(len(points)), "consensus_volume_mm3": len(points) * voxel_volume, "mask_voxels_outside_crop": int(len(points) - nodule.sum()),
            "center_x_mm": center_lps[0], "center_y_mm": center_lps[1], "center_z_mm": center_lps[2], "pixel_spacing_mm": float(affine[0, 0]), "slice_spacing_mm": float(affine[2, 2]), "slice_thickness_mm": float(scan.slice_thickness),
            **{name: float(frame[name].median()) for name in config["ratings"]["ordinal"]}, "calcification": float((frame.calcification < 6).mean()),
            **{f"contact_{key}": value for key, value in contact_measures(image, nodule.astype(bool), crop_affine, contact_settings).items()},
            "image": str(paths["image"].relative_to(output)), "mask": str(paths["mask"].relative_to(output))})
    return receipt


def nodules(workers, limit):
    config = configuration()
    lidc_root, output, _, _ = directories()
    for name in ["crops", "masks", "scans"]:
        (output / name).mkdir(parents=True, exist_ok=True)
    scan_ids = sorted(scan.id for scan in pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= config["nodules"]["maximum_slice_thickness_mm"]))
    if limit:
        scan_ids = scan_ids[:: max(1, len(scan_ids) // limit)][:limit]
    pending = [scan_id for scan_id in scan_ids if not (output / "scans" / f"{scan_id}.json").exists()]
    print(f"LIDC nodules: {len(scan_ids)} scans with slice thickness <= {config['nodules']['maximum_slice_thickness_mm']} mm, {len(pending)} pending, {workers} workers", flush=True)
    started = time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(process_scan, scan_id, lidc_root, output) for scan_id in pending]
        for count, future in enumerate(as_completed(futures), 1):
            receipt = future.result()
            write_json(output / "scans" / f"{receipt['scan_id']}.json", receipt)
            if count % 20 == 0 or count == len(pending):
                rate = count / (time.monotonic() - started)
                print(f"LIDC nodules: {count}/{len(pending)} scans, {rate:.2f} scans/s, eta {(len(pending) - count) / rate / 60:.1f} min", flush=True)
    receipts = [read_json(output / "scans" / f"{scan_id}.json") for scan_id in scan_ids]
    table = pd.DataFrame([row for receipt in receipts for row in receipt["nodules"]])
    readers = pd.DataFrame([row for receipt in receipts for row in receipt["readers"]])
    exclusions = pd.DataFrame([row for receipt in receipts for row in receipt["exclusions"]], columns=["scan_id", "cluster", "reason"])
    suffix = f"_smoke{limit}" if limit else ""
    table.to_csv(output / f"nodules{suffix}.csv", index=False)
    readers.to_csv(output / f"readers{suffix}.csv", index=False)
    exclusions.to_csv(output / f"exclusions{suffix}.csv", index=False)
    members_table = select_members(table, config)
    members_table.to_csv(output / f"members{suffix}.csv", index=False)
    write_json(output / f"nodules{suffix}.json", {
        "run": config["run_id"], "scans": len(scan_ids), "scans_with_eligible_nodules": int(sum(receipt["clusters_with_enough_readers"] > 0 for receipt in receipts)),
        "scans_excluded": int(exclusions.cluster.isna().sum()), "exclusion_reasons": exclusions.reason.value_counts().to_dict(), "nodules": len(table), "patients": int(table.PatientID.nunique()),
        "members": len(members_table), "halves": members_table.split.value_counts().to_dict(), "readers_per_member": members_table.readers.value_counts().to_dict(),
        "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(Path(__file__)), "created_at": timestamp()})
    print(f"LIDC nodules: {len(table)} nodules from {table.PatientID.nunique()} patients, {len(members_table)} members, excluded scans {int(exclusions.cluster.isna().sum())}", flush=True)


def select_members(table, config):
    ordered = table.sort_values(["PatientID", "readers", "consensus_volume_mm3", "scan_id", "cluster"], ascending=[True, False, False, True, True])
    members_table = ordered.groupby("PatientID", sort=True).head(1).copy()
    order = members_table.PatientID.map(lambda value: hashlib.sha256(f"lidc-half:{value}".encode()).hexdigest())
    members_table = members_table.assign(order=order).sort_values("order")
    half = len(members_table) // 2
    members_table["split"] = ["discovery"] * half + ["confirmation"] * (len(members_table) - half)
    return members_table.drop(columns="order").sort_values("identifier").reset_index(drop=True)


def check_case(values, previous, campaign, model):
    trained_key, random_key, distance_key = SITE_ARRAYS[model]
    return {"trained_max_abs_error": float(np.abs(values[trained_key] - previous[trained_key]).max()), "random_max_abs_error": float(np.abs(values[random_key] - campaign[random_key]).max()),
            "distance_max_abs_error": float(np.abs(values[distance_key] - campaign[distance_key]).max()), "trained_scale": float(np.abs(previous[trained_key]).max())}


def encode(device, verify, smoke):
    config, campaign = configuration(), campaign_configuration()
    _, output, run, campaign_root = directories()
    storage = context()[0]
    previous = Path(storage["activations"]) / campaign["previous_suite"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoders = {}
    for name in ["fmcib", "vista"]:
        randomized = random_encoder(name, campaign["random_encoder"]["initialization_seed"])
        assert parameter_digest(randomized) == campaign["whole_ct"]["random_encoder_sha256"][name], name
        encoders[name] = (load_encoder(name)[0].to(device).eval(), randomized.to(device))
    if verify:
        frame = pd.read_csv(campaign_root / "members.csv", dtype={"PatientID": str}).sort_values("identifier").iloc[:: 97][:verify]
        destination = run / "verification"
    else:
        frame = pd.read_csv(output / (f"members_smoke{smoke}.csv" if smoke else "members.csv"), dtype={"PatientID": str})
        frame = frame.assign(image=frame["image"].map(lambda value: str(output / value)), mask=frame["mask"].map(lambda value: str(output / value)), label=-1)
        destination = run / ("cases_smoke" if smoke else "cases")
    for model in encoders:
        (destination / model).mkdir(parents=True, exist_ok=True)
    pending = frame[[not (destination / "vista" / f"{value}.json").exists() for value in frame.identifier]]
    print(f"ENCODE {len(pending)} of {len(frame)} records pending ({'LUNA25 verification' if verify else 'LIDC'})", flush=True)
    loader = torch.utils.data.DataLoader(RecordDataset(pending, campaign, []), batch_size=None, collate_fn=unchanged, num_workers=6, persistent_workers=False, prefetch_factor=4)
    started, checks = time.monotonic(), []
    for count, item in enumerate(loader, 1):
        identifier = item["identifier"]
        with torch.inference_mode():
            deployed = torch.from_numpy(item["deployed"])[None, None].to(device)
            window = torch.from_numpy(item["vista"])[None, None].to(device)
            values = {
                "fmcib": {"layer2": fmcib_prefix(encoders["fmcib"][0], deployed, "layer2")[0].cpu().numpy(), "layer2_random": fmcib_prefix(encoders["fmcib"][1], deployed, "layer2")[0].cpu().numpy(),
                          "layer2_distance": item["arrays"]["fmcib"]["layer2_distance"]},
                "vista": {"stage2": vista_sites(encoders["vista"][0], window)["stage2"][0].cpu().numpy(), "stage2_random": vista_sites(encoders["vista"][1], window)["stage2"][0].cpu().numpy(),
                          "stage2_distance": item["arrays"]["vista"]["stage2_distance"]},
            }
        for model, arrays in values.items():
            assert all(np.isfinite(value).all() for key, value in arrays.items() if not key.endswith("distance")), (identifier, model)
            if verify:
                with np.load(io.BytesIO((previous / model / "cases" / f"{identifier}.npz").read_bytes())) as old, np.load(io.BytesIO((campaign_root / "cases" / model / f"{identifier}.npz").read_bytes())) as new:
                    checks.append({"identifier": identifier, "model": model, **check_case(arrays, old, new, model)})
            else:
                path = destination / model / f"{identifier}.npz"
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **arrays)
                temporary.replace(path)
            write_json((destination / model / identifier).with_suffix(".json"), item["geometry"])
        if count % 50 == 0 or count == len(pending):
            rate = count / (time.monotonic() - started)
            print(f"ENCODE {count}/{len(pending)} records, {rate:.2f} records/s, eta {(len(pending) - count) / rate / 60:.1f} min", flush=True)
    if verify:
        table = pd.DataFrame(checks)
        table.to_csv(destination / "luna25_checks.csv", index=False)
        summary = {model: rows.drop(columns=["identifier", "model"]).max().to_dict() for model, rows in table.groupby("model")}
        write_json(destination / "luna25_checks.json", {"records": int(table.identifier.nunique()), "maximum": summary, "created_at": timestamp()})
        print(summary, flush=True)
        for model, row in summary.items():
            assert row["trained_max_abs_error"] < 1e-3 * max(row["trained_scale"], 1.0) and row["random_max_abs_error"] < 1e-3 * max(row["trained_scale"], 1.0) and row["distance_max_abs_error"] < 1e-4, (model, row)


def region_statistics(codes, distance):
    within, near = distance <= 4, (distance > 4) & (distance <= 10)
    empty = torch.full((codes.shape[1],), float("nan"), device=codes.device)
    return torch.stack([codes[within].mean(0) if within.any() else empty, codes[near].mean(0) if near.any() else empty, codes[within].amax(0) if within.any() else empty], -1)


@torch.inference_mode()
def activations(device, smoke):
    config, campaign = configuration(), campaign_configuration()
    _, output, run, campaign_root = directories()
    frame = pd.read_csv(output / (f"members_smoke{smoke}.csv" if smoke else "members.csv"), dtype={"PatientID": str})
    cases = run / ("cases_smoke" if smoke else "cases")
    for name in config["sites"]:
        model, site = name.split(":")
        destination = run / f"activations{'_smoke' if smoke else ''}_{model}_{site}.npz"
        if destination.exists():
            continue
        trained, randomized, others = dictionaries(campaign_root, model, site, config["reference_seed"])
        stable, matched, seeds = stable_mask(campaign_root, trained["stability_group"], campaign)
        assert seeds[0] == config["reference_seed"]
        load = lambda row: SpatialDictionary(campaign_root / "runs" / row["run_id"] / "dictionary.pt", device=device)
        trained_sae, random_sae = load(trained), load(randomized)
        stable_index = np.flatnonzero(stable)
        other_saes = {int(seed): (load(others[int(seed)]), torch.from_numpy(matched[0, position, stable_index].astype(np.int64)).to(device)) for position, seed in enumerate(seeds) if position > 0}
        trained_key, random_key, distance_key = SITE_ARRAYS[model]
        result = {"trained": [], "random": [], "matched": {seed: [] for seed in other_saes}}
        for identifier in frame.identifier:
            with np.load(cases / model / f"{identifier}.npz") as saved:
                trained_values, random_values, distance = saved[trained_key], saved[random_key], saved[distance_key]
            keep = np.flatnonzero(distance.reshape(-1) <= 10)
            distance = torch.from_numpy(distance.reshape(-1)[keep]).to(device)
            tokens = torch.from_numpy(trained_values.reshape(trained_values.shape[0], -1).T[keep]).to(device)
            random_tokens = torch.from_numpy(random_values.reshape(random_values.shape[0], -1).T[keep]).to(device)
            result["trained"].append(region_statistics(trained_sae.encode(tokens), distance).cpu().numpy())
            result["random"].append(region_statistics(random_sae.encode(random_tokens), distance).cpu().numpy())
            for seed, (sae, columns) in other_saes.items():
                result["matched"][seed].append(region_statistics(sae.encode(tokens)[:, columns], distance).cpu().numpy())
        np.savez_compressed(destination, identifiers=frame.identifier.to_numpy(dtype=str), trained=np.stack(result["trained"]).astype(np.float32), random=np.stack(result["random"]).astype(np.float32),
                            matched=np.stack([np.stack(result["matched"][seed]) for seed in sorted(result["matched"])]).astype(np.float32), matched_seeds=np.asarray(sorted(result["matched"])),
                            stable=stable, random_stable=stable_mask(campaign_root, randomized["stability_group"], campaign)[0], statistics=np.asarray(config["activation"]["statistics"]))
        write_json(destination.with_suffix(".json"), {"site": name, "trained_run": trained["run_id"], "random_run": randomized["run_id"], "matched_runs": {int(seed): others[int(seed)]["run_id"] for seed in other_saes},
                                                     "stable_features": int(stable.sum()), "nodules": len(frame), "created_at": timestamp()})
        print("ACTIVATIONS", name, len(frame), "nodules", int(stable.sum()), "stable features", flush=True)


def partial(values, ratings, size):
    # Partial Spearman: Pearson correlation of the rank residuals (no re-ranking of the residuals).
    rho = correlate(residual_ranks(values, size), residual_ranks(ratings, size))
    freedom = len(values) - 3
    t = rho * np.sqrt(freedom / np.clip(1 - rho * rho, 1e-12, None))
    return rho, 2 * stats.t.sf(np.abs(t), freedom)


def holm(p):
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(np.asarray(p)[order] * (len(p) - np.arange(len(p)))))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def ceiling(members_table, readers, ratings, size, config):
    settings = config["ceiling"]
    generator = np.random.default_rng(settings["seed"])
    grouped = {identifier: rows for identifier, rows in readers.groupby("identifier")}
    rows = []
    for rating in ratings:
        raw, adjusted = [], []
        for _ in range(settings["repeats"]):
            drawn, others = [], []
            for identifier in members_table.identifier:
                values = grouped[identifier][rating].to_numpy()
                if rating == "calcification":
                    values = (values < 6).astype(float)
                pick = generator.integers(len(values))
                rest = np.delete(values, pick)
                drawn.append(values[pick])
                others.append(rest.mean() if rating == "calcification" else np.median(rest))
            drawn, others = np.asarray(drawn, dtype=float)[:, None], np.asarray(others, dtype=float)[:, None]
            raw.append(spearman_matrix(drawn, others)[0, 0])
            adjusted.append(partial(drawn, others, size)[0][0, 0])
        rows.append({"rating": rating, "reader_vs_others_rho": float(np.nanmean(raw)), "reader_vs_others_size_adjusted_rho": float(np.nanmean(adjusted))})
    return pd.DataFrame(rows)


def analyze(smoke):
    config = configuration()
    settings = config["association"]
    _, output, run, _ = directories()
    storage = context()[0]
    suffix = f"_smoke{smoke}" if smoke else ""
    members_table = pd.read_csv(output / f"members{suffix}.csv", dtype={"PatientID": str})
    readers = pd.read_csv(output / f"readers{suffix}.csv")
    ratings = config["ratings"]["ordinal"] + ["calcification"]
    radiomics = Path(storage["runs"]) / config["radiomics_run"] / "analysis"
    explanations = pd.read_csv(radiomics / "explanations.csv")
    effects = pd.read_csv(radiomics / "trained_features.csv")[["site", "feature", "effect_rank", "contribution_mean_abs", "category", "development_r2"]]
    statistics = config["activation"]["statistics"]
    destination = run / f"analysis{suffix}"
    destination.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(settings["bootstrap_seed"])
    tables, prespecified, summaries = [], [], {}
    for name in config["sites"]:
        model, site = name.split(":")
        with np.load(run / f"activations{'_smoke' if smoke else ''}_{model}_{site}.npz") as saved:
            assert (saved["identifiers"] == members_table.identifier.to_numpy()).all()
            arrays = {key: saved[key] for key in saved.files}
        # Region emptiness depends only on the grid distance map, so one feature decides it for the whole site.
        valid = np.isfinite(arrays["trained"][:, 0, :2]).all(1)
        frame = members_table[valid].reset_index(drop=True)
        halves = {half: (frame.split == half).to_numpy() for half in ["discovery", "confirmation"]}
        size = np.log(frame.consensus_volume_mm3.to_numpy())[:, None]
        values_ratings = frame[ratings].to_numpy(dtype=float)
        targets, features, peaks = {}, {}, {}
        for encoder, mask_key in [("trained", "stable"), ("random", "random_stable")]:
            features[encoder] = np.flatnonzero(arrays[mask_key])
            rows = explanations[(explanations.site == name) & (explanations.encoder == encoder)].set_index("feature")
            peaks[encoder] = rows.loc[features[encoder], "peak_region"].map(statistics.index).to_numpy()
            targets[encoder] = arrays[encoder][valid][:, features[encoder], peaks[encoder]]
        matched = arrays["matched"][:, valid][:, :, np.arange(len(features["trained"])), peaks["trained"]]
        correlations = {}
        for encoder in ["trained", "random"]:
            active = np.stack([(targets[encoder][halves[half]] > 0).sum(0) for half in halves], 1).min(1) >= settings["minimum_active"]
            correlations[encoder] = {"active": active, **{half: partial(targets[encoder][halves[half]], values_ratings[halves[half]], size[halves[half]]) for half in halves}}
        rows = []
        for index, rating in enumerate(ratings):
            threshold = float(np.nanpercentile(np.abs(correlations["random"]["discovery"][0][correlations["random"]["active"], index]), 95))
            for encoder in ["trained", "random"]:
                result = correlations[encoder]
                for position, feature in enumerate(features[encoder]):
                    discovery_rho = result["discovery"][0][position, index]
                    rows.append({"site": name, "encoder": encoder, "rating": rating, "feature": int(feature), "active": bool(result["active"][position]),
                                 "active_discovery": int((targets[encoder][halves["discovery"], position] > 0).sum()), "active_confirmation": int((targets[encoder][halves["confirmation"], position] > 0).sum()),
                                 "discovery_rho": discovery_rho, "discovery_p": result["discovery"][1][position, index], "random_threshold": threshold,
                                 "selected": bool(result["active"][position] and abs(np.nan_to_num(discovery_rho)) > threshold),
                                 "confirmation_rho": result["confirmation"][0][position, index], "confirmation_p": result["confirmation"][1][position, index]})
        table = pd.DataFrame(rows).assign(confirmation_q=np.nan, confirmed=False, replicated_seeds=np.nan, confirmation_low=np.nan, confirmation_high=np.nan)
        for encoder in ["trained", "random"]:
            selected = (table.encoder == encoder) & table.selected
            if selected.any():
                # A confirmation value without variance counts as p = 1.
                q = stats.false_discovery_control(table.loc[selected, "confirmation_p"].fillna(1.0).to_numpy(), method="bh")
                same = (np.sign(table.loc[selected, "confirmation_rho"].fillna(0.0)) == np.sign(table.loc[selected, "discovery_rho"])).to_numpy()
                table.loc[selected, "confirmation_q"] = q
                table.loc[selected, "confirmed"] = same & (q < 0.05)
        position_of = {int(feature): position for position, feature in enumerate(features["trained"])}
        confirmation = halves["confirmation"]
        for row_index in table.index[(table.encoder == "trained") & table.confirmed]:
            row = table.loc[row_index]
            position, index = position_of[int(row.feature)], ratings.index(row.rating)
            rating_values, covariate = values_ratings[confirmation][:, [index]], size[confirmation]
            replicated = 0
            for seed_values in matched:
                rho, p = partial(seed_values[confirmation][:, [position]], rating_values, covariate)
                replicated += int(np.sign(rho[0, 0]) == np.sign(row.confirmation_rho) and p[0, 0] < 0.05)
            target = targets["trained"][confirmation][:, [position]]
            draws = []
            for _ in range(settings["bootstrap"]):
                sample = generator.integers(len(target), size=len(target))
                draws.append(partial(target[sample], rating_values[sample], covariate[sample])[0][0, 0])
            table.loc[row_index, ["replicated_seeds", "confirmation_low", "confirmation_high"]] = [replicated, *np.nanpercentile(draws, [2.5, 97.5])]
        table = table.merge(effects[effects.site == name].drop(columns="site"), on="feature", how="left")
        table.loc[table.encoder == "random", ["effect_rank", "contribution_mean_abs", "category", "development_r2"]] = np.nan
        table["clear"] = (table.encoder == "trained") & table.confirmed & (table.replicated_seeds >= 6) & table.active & (table.effect_rank <= config["classifier_effect"]["effect_rank"])
        tables.append(table)
        summaries[name] = {"nodules": int(valid.sum()), "excluded_empty_region": int((~valid).sum()), "halves": {half: int(mask.sum()) for half, mask in halves.items()},
                           "stable": {encoder: int(len(features[encoder])) for encoder in features}, "active": {encoder: int(correlations[encoder]["active"].sum()) for encoder in correlations},
                           "selected_pairs": table[table.selected].groupby("encoder").size().to_dict(), "confirmed_pairs": table[table.confirmed].groupby("encoder").size().to_dict(),
                           "confirmed_pairs_per_active_feature": {encoder: float(((table.encoder == encoder) & table.confirmed).sum() / correlations[encoder]["active"].sum()) for encoder in correlations},
                           "clear_features": sorted(int(value) for value in table[table.clear].feature.unique())}
        measures = {rating: values_ratings[:, ratings.index(rating)] for rating in ratings} | {measure: frame[f"contact_{measure}"].to_numpy(dtype=float) for measure in config["contact"]["measures"]}
        for feature in config["prespecified"]["features"][name]:
            position = position_of[feature]
            hypothesis = config["prespecified"]["hypotheses"].get(f"{name}:{feature}", {})
            for measure, value in measures.items():
                rho, p = partial(targets["trained"][:, [position]], value[:, None], size)
                stated = hypothesis.get("measure") == measure
                prespecified.append({"site": name, "feature": feature, "measure": measure, "rho": rho[0, 0], "p": p[0, 0], "nodules": int(valid.sum()), "active": int((targets["trained"][:, position] > 0).sum()),
                                     "hypothesis_sign": hypothesis["sign"] if stated else np.nan, "luna25_rho": hypothesis["luna25_size_adjusted_rho"] if stated else np.nan})
    associations = pd.concat(tables, ignore_index=True)
    associations.to_csv(destination / "associations.csv", index=False)
    prespecified = pd.DataFrame(prespecified)
    prespecified["p_holm"] = holm(prespecified.p.to_numpy())
    prespecified.to_csv(destination / "prespecified.csv", index=False)
    size_all = np.log(members_table.consensus_volume_mm3.to_numpy())[:, None]
    agreement = ceiling(members_table, readers, ratings, size_all, config)
    agreement.to_csv(destination / "ceiling.csv", index=False)
    clear = associations[associations.clear].sort_values(["site", "effect_rank", "rating"])
    clear.to_csv(destination / "clear_features.csv", index=False)
    ceilings = agreement.set_index("rating").reader_vs_others_size_adjusted_rho
    clear_summary = clear.assign(term=lambda frame: frame.apply(lambda row: f"{row.rating} {row.confirmation_rho:+.2f} [{row.confirmation_low:+.2f}, {row.confirmation_high:+.2f}], {abs(row.confirmation_rho) / ceilings[row.rating]:.0%} of one radiologist", axis=1))
    clear_summary = clear_summary.groupby(["site", "feature"], sort=False).agg(effect_rank=("effect_rank", "first"), active_discovery=("active_discovery", "first"), active_confirmation=("active_confirmation", "first"),
                                                                                 luna25_radiomics_category=("category", "first"), luna25_radiomics_r2=("development_r2", "first"), confirmed_ratings=("term", "; ".join)).reset_index()
    clear_summary.to_csv(destination / "clear_summary.csv", index=False)
    write_json(destination / "summary.json", {"sites": summaries, "ceiling": agreement.set_index("rating").to_dict("index"), "members": len(members_table),
                                              "rating_distribution": {rating: members_table[rating].describe().round(3).to_dict() for rating in ratings},
                                              "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    figure(associations, agreement, ratings, destination)
    prespecified_figure(prespecified, config, destination)
    print(pd.DataFrame(summaries).T.to_string(), flush=True)
    print(clear[["site", "feature", "rating", "confirmation_rho", "replicated_seeds", "effect_rank"]].to_string(index=False), flush=True)


def figure(associations, agreement, ratings, destination):
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    colors = {"trained": "#2a78d6", "random": "#eb6834"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    sites = list(dict.fromkeys(associations.site))
    fig, axes = plt.subplots(len(sites), 1, figsize=(9, 3.6 * len(sites)), facecolor=surface, squeeze=False)
    ceiling_values = agreement.set_index("rating").reader_vs_others_size_adjusted_rho
    for axis, site in zip(axes[:, 0], sites):
        for offset, encoder in [(-0.17, "trained"), (0.17, "random")]:
            rows = associations[(associations.site == site) & (associations.encoder == encoder) & associations.active]
            for index, rating in enumerate(ratings):
                values = np.abs(rows[rows.rating == rating].confirmation_rho.dropna().to_numpy())
                jitter = np.random.default_rng(index).uniform(-0.07, 0.07, len(values))
                axis.scatter(index + offset + jitter, values, s=6, color=colors[encoder], alpha=0.35, linewidth=0, label=f"{encoder} encoder, analysed stable features" if index == 0 else None)
                confirmed = rows[(rows.rating == rating) & rows.confirmed]
                axis.scatter(np.full(len(confirmed), index + offset), np.abs(confirmed.confirmation_rho), s=34, facecolor="none", edgecolor=colors[encoder], linewidth=1.2,
                             label=f"{encoder} encoder, confirmed" if index == 0 else None, zorder=3)
        axis.hlines([abs(ceiling_values[rating]) for rating in ratings], np.arange(len(ratings)) - 0.35, np.arange(len(ratings)) + 0.35, color=ink, linewidth=2, label="one radiologist vs the other radiologists (size-adjusted)", zorder=4)
        axis.set_xticks(np.arange(len(ratings)), ratings)
        axis.set_ylabel("|size-adjusted partial Spearman|\nconfirmation half")
        axis.set_title(site, loc="left")
        axis.set_facecolor(surface)
        axis.grid(axis="y", color=grid, linewidth=0.6)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"ratings.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


def prespecified_figure(prespecified, config, destination):
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    prespecified = prespecified.assign(label=prespecified.site.str.replace(":", " ") + " " + prespecified.feature.astype(str))
    labels, measures = list(dict.fromkeys(prespecified.label)), list(dict.fromkeys(prespecified.measure))
    matrix = prespecified.pivot(index="label", columns="measure", values="rho").loc[labels, measures].to_numpy()
    significant = prespecified.pivot(index="label", columns="measure", values="p_holm").loc[labels, measures].to_numpy() < 0.05
    limit = max(0.3, float(np.nanmax(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(1.0 + 0.62 * len(measures), 1.2 + 0.42 * len(labels)), facecolor=surface)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(len(labels)):
        for column in range(len(measures)):
            axis.text(column, row, f"{matrix[row, column]:+.2f}" + ("*" if significant[row, column] else ""), ha="center", va="center", fontsize=7, color=ink)
    axis.set_xticks(np.arange(len(measures)), measures, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_title("review-list features on LIDC: size-adjusted partial Spearman (* Holm p < 0.05)", loc="left")
    fig.colorbar(image, ax=axis, fraction=0.03)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"prespecified.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["nodules", "encode", "activations", "analyze"], required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--smoke", type=int, help="number of scans for a smoke run; outputs carry a smoke suffix")
    parser.add_argument("--verify", type=int, help="encode stage: recompute this many LUNA25 records and compare with the cached campaign arrays")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.stage == "nodules":
        nodules(args.workers, args.smoke)
    elif args.stage == "encode":
        encode(args.device, args.verify, args.smoke)
    elif args.stage == "activations":
        activations(args.device, args.smoke)
    else:
        analyze(args.smoke)


if __name__ == "__main__":
    main()

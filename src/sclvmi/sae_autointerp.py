import argparse
import multiprocessing
import time
from functools import lru_cache
from pathlib import Path

import h5py
import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_context import configuration as campaign_configuration
from .sae_context import prepare_record, stable_value
from .sae_interp_profiles import load_case, profile_members


CONFIG_PATH = ROOT / "configs/sae_autointerp.json"
SOURCES = {"fmcib:layer2": "fmcib", "vista:stage2": "vista"}
GRID_AFFINES = {"fmcib:layer2": "layer2_grid_affine_ras", "vista:stage2": "stage2_grid_affine_ras"}
REFERENCES = ["sphericity", "hu_mean", "hu_std", "hu_p10", "hu_p90", "solid_fraction", "ring_0_5mm_hu_mean", "ring_5_10mm_hu_mean"]


def paths():
    config = read_json(CONFIG_PATH)
    storage = context()[0]
    runs = Path(storage["runs"])
    campaign = campaign_configuration()
    assert campaign["campaign"] == config["campaign"]
    return config, runs / config["campaign"], Path(storage["activations"]) / campaign["previous_suite"], runs / config["candidates_run"], runs / config["run_id"]


def case_frame(root):
    members = profile_members(root)
    assert set(members.split) == {"train", "development"}
    references = pd.read_csv(root / "nodule_references.csv", dtype={"PatientID": str}).set_index("identifier").loc[members.identifier]
    members = members.assign(log_volume=np.log(references.volume_mm3.to_numpy()))
    return members.reset_index(drop=True), references.reset_index()


def choose_features(config, candidates, members):
    rules = config["features"]
    development = (members.split == "development").to_numpy()
    chosen = []
    for site in config["sites"]:
        name = site.replace(":", "_")
        with np.load(candidates / name / "summaries.npz") as saved:
            active = {encoder: (saved[encoder][development] > 0).sum(0) for encoder in ["trained", "random"]}
        request = read_json(candidates / name / "summary.json")["runs"]
        tables = {}
        for encoder in ["trained", "random"]:
            table = pd.read_csv(candidates / name / f"{encoder}_features.csv")
            tables[encoder] = table.assign(active=active[encoder][table.feature.to_numpy()])
        trained, random = tables["trained"], tables["random"]
        usable = trained.active >= rules["minimum_active_confirmation"]
        picked = []
        families = set()
        for row in trained[trained.confirmed & usable].sort_values("delta_auroc", ascending=False).itertuples():
            if row.family not in families and len(picked) < rules["candidates_per_site"]:
                families.add(row.family)
                picked.append(("trained", "candidate", int(row.feature)))
        nodule = trained[trained.stable & ~trained.confirmed & usable & (trained.nodule_enrichment >= rules["enrichment_minimum"])].copy()
        nodule["peak_region"] = nodule[["region_inside", "region_margin_0_4mm", "region_near_4_10mm"]].to_numpy().argmax(1)
        queues = [list(nodule[nodule.peak_region == region].sort_values("nodule_enrichment", ascending=False).feature) for region in range(3)]
        taken = 0
        while taken < rules["nodule_per_site"]:
            assert any(queues), f"{site}: too few nodule-enriched stable features"
            for queue in queues:
                if queue and taken < rules["nodule_per_site"]:
                    picked.append(("trained", "nodule", int(queue.pop(0))))
                    taken += 1
        controls = random[random.stable & (random.active >= rules["minimum_active_confirmation"])].sort_values(["confirmed", "nodule_enrichment"], ascending=[False, False])
        assert len(controls) >= rules["random_per_site"], f"{site}: too few active stable random-weight features"
        picked += [("random", "random", int(feature)) for feature in controls.feature[:rules["random_per_site"]]]
        picked += [("trained", "shuffled", picked[0][2])]
        for encoder, role, feature in picked:
            row = tables[encoder].set_index("feature").loc[feature]
            chosen.append({"key": f"{name}_{role}_{feature}", "site": site, "encoder": encoder, "role": role, "feature": feature, "run_id": request[encoder],
                           "nodule_enrichment": float(row.nodule_enrichment), "confirmed": bool(row.confirmed), "delta_auroc": float(row.delta_auroc) if np.isfinite(row.delta_auroc) else None,
                           "partial_confirmation": float(row.partial_confirmation) if np.isfinite(row.partial_confirmation) else None, "active_confirmation": int(row.active),
                           "regions": {band: float(row[f"region_{band}"]) for band in ["inside", "margin_0_4mm", "near_4_10mm", "far_over_10mm"]}, "family": None if pd.isna(row.family) else int(row.family)})
    return chosen


def feature_values(config, candidates, feature, members):
    with np.load(candidates / feature["site"].replace(":", "_") / "summaries.npz") as saved:
        values = saved[feature["encoder"]][:, feature["feature"]].astype(np.float64)
    if feature["role"] == "shuffled":
        generator = np.random.default_rng([config["features"]["shuffle_seed"], stable_value(feature["key"]) % 2**32])
        for split in ["train", "development"]:
            rows = np.flatnonzero((members.split == split).to_numpy())
            values[rows] = values[generator.permutation(rows)]
    return values


def pick_examples(values, members, split, high_count, low_count, quantile, prefix):
    rows = np.flatnonzero((members.split == split).to_numpy())
    order = sorted(rows, key=lambda row: (-values[row], stable_value(members.identifier[row])))
    high = order[:high_count]
    assert values[high].min() > 0, "fewer active patients than high examples"
    threshold = np.quantile(values[rows], quantile)
    taken = set(high)
    pool = [row for row in rows if values[row] <= threshold and row not in taken]
    low = []
    for row in high[:low_count]:
        nearest = min(pool, key=lambda candidate: (abs(members.log_volume[candidate] - members.log_volume[row]), stable_value(members.identifier[candidate])))
        pool.remove(nearest)
        low.append(nearest)
    records = [(f"{prefix}H{index + 1:02d}", "high", row) for index, row in enumerate(high)] + [(f"{prefix}L{index + 1:02d}", "low", row) for index, row in enumerate(low)]
    return [{"example_id": example, "group": group, "row": int(row), "identifier": members.identifier[row], "PatientID": members.PatientID[row], "value": float(values[row]),
             "log_volume": float(members.log_volume[row])} for example, group, row in records], float(threshold)


def select(config, root, candidates, output):
    members, _ = case_frame(root)
    features = choose_features(config, candidates, members)
    rules = config["examples"]
    rows = []
    for feature in features:
        values = feature_values(config, candidates, feature, members)
        generation, generation_threshold = pick_examples(values, members, "train", rules["generation_high"], rules["generation_low"], rules["low_quantile"], "")
        validation, validation_threshold = pick_examples(values, members, "development", rules["validation_high"], rules["validation_low"], rules["low_quantile"], "V")
        feature["low_thresholds"] = {"generation": generation_threshold, "validation": validation_threshold}
        rows += [{**item, "key": feature["key"], "stage": "generation"} for item in generation] + [{**item, "key": feature["key"], "stage": "validation"} for item in validation]
    examples = pd.DataFrame(rows)
    matched = examples.assign(pair=examples.example_id.str[-2:]).pivot_table(index=["key", "stage", "pair"], columns="group", values="log_volume").dropna()
    output.mkdir(parents=True, exist_ok=True)
    examples.to_csv(output / "examples.csv", index=False)
    write_json(output / "features.json", {"features": features, "size_matching_mean_abs_log_volume": float((matched.high - matched.low).abs().mean()), "created_at": timestamp(), "test_used": False})
    write_json(output / "request.json", {"config": config, "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    print("SELECTED", len(features), "features", len(examples), "examples", examples.identifier.nunique(), "patients", flush=True)
    for feature in features:
        print(" ", feature["key"], feature["role"], "enrichment", round(feature["nodule_enrichment"], 2), "active", feature["active_confirmation"], flush=True)


def measure(config, root, candidates, output):
    # Baseline: how well measured nodule properties alone separate the high and low validation patients of each feature.
    members, references = case_frame(root)
    design = np.column_stack([members.log_volume.to_numpy(), references[REFERENCES].to_numpy(dtype=np.float64)])
    usable = np.isfinite(design).all(1)
    train = (members.split == "train").to_numpy() & usable
    examples = pd.read_csv(output / "examples.csv", dtype={"PatientID": str})
    rows = []
    for feature in read_json(output / "features.json")["features"]:
        values = feature_values(config, candidates, feature, members)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))).fit(design[train], pd.Series(values[train]).rank().to_numpy())
        validation = examples[(examples.key == feature["key"]) & (examples.stage == "validation")]
        predicted = np.where(usable[validation.row.to_numpy()], model.predict(np.nan_to_num(design[validation.row.to_numpy()])), np.nan)
        rows += [{"key": feature["key"], "example_id": example, "measurement_score": score} for example, score in zip(validation.example_id, predicted)]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "measurement.csv", index=False)
    print("MEASURED", frame.key.nunique(), "features;", int(frame.measurement_score.isna().sum()), "validation examples without complete measurements; training patients used", int(train.sum()), flush=True)


def input_path(root, output, source, identifier):
    viewer = root / "viewer_inputs" / source / f"{identifier}.h5"
    return viewer if viewer.exists() else output / "inputs" / source / f"{identifier}.h5"


def prepare_inputs(task):
    identifier, sources, root, output, threads = task
    torch.set_num_threads(threads)
    members = pd.read_csv(root / "members.csv", dtype={"PatientID": str}).set_index("identifier")
    row = next(members.loc[[identifier]].reset_index().itertuples())
    record = prepare_record(row, campaign_configuration(), True)
    for source in sources:
        view = record["views"][source]
        destination = output / "inputs" / source / f"{identifier}.h5"
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".h5.part")
        with h5py.File(partial, "w") as saved:
            saved.create_dataset("input_hu", data=view["input_hu"], compression="gzip", compression_opts=1)
            saved.create_dataset("mask", data=view["mask"], compression="gzip", compression_opts=1)
            saved.attrs["canonical_affine_ras"] = view["canonical_affine_ras"]
            saved.attrs["input_affine_ras"] = view["input_affine_ras"]
            saved.attrs["identifier"] = identifier
        partial.replace(destination)
    return identifier


def load_volume(path):
    with h5py.File(path, "r") as saved:
        return saved["input_hu"][...].astype(np.float32), saved["mask"][...] > 0, np.asarray(saved.attrs["canonical_affine_ras"])


@lru_cache(maxsize=4)
def dictionary(path):
    return SpatialDictionary(path, device="cpu")


def feature_map(root, previous, feature, identifier, canonical_affine, shape):
    model, site = feature["site"].split(":")
    trained, random, distance, _, valid = load_case(root, previous, model, site, identifier)
    values = trained if feature["encoder"] == "trained" else random
    sae = dictionary(root / "runs" / feature["run_id"] / "dictionary.pt")
    tokens = torch.from_numpy(values).flatten(1).T
    with torch.inference_mode():
        normalized = (tokens - sae.mean) / sae.scale - sae.weights["b_dec"]
        activity = torch.relu(normalized @ sae.weights["W_enc"][:, feature["feature"]])
        activity = (activity * (activity > sae.threshold)).numpy().reshape(values.shape[1:])
    activity = np.where(valid, activity, 0)
    near = (distance <= 4.0) & valid
    case_value = float(activity[near].max()) if near.any() else 0.0
    grid = np.asarray(read_json(root / "cases" / "fmcib" / f"{identifier}.json")[GRID_AFFINES[feature["site"]]])
    native_to_canonical = np.linalg.inv(canonical_affine) @ grid
    coordinates = np.indices(shape, dtype=np.float64).reshape(3, -1).T
    position = nib.affines.apply_affine(np.linalg.inv(native_to_canonical), coordinates)
    mapped = map_coordinates(activity, position.T, order=1, mode="nearest").reshape(shape)
    return mapped.astype(np.float32), case_value


def planes(volume, center):
    x, y, z = center
    # Radiological orientation: patient right on the left, anterior and superior at the top.
    return [volume[:, :, z][::-1, ::-1].T, volume[:, y, :][::-1, ::-1].T, volume[x, :, :][::-1, ::-1].T]


def plane_spacing(spacing):
    return [(spacing[1], spacing[0]), (spacing[2], spacing[0]), (spacing[2], spacing[1])]


def fit(image, row_mm, column_mm, size, resample):
    height, width = image.shape[0] * row_mm, image.shape[1] * column_mm
    scale = size / max(height, width)
    resized = Image.fromarray(np.ascontiguousarray(image)).resize((round(width * scale), round(height * scale)), resample)
    canvas = Image.new(resized.mode, (size, size))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas, scale


def windowed(values, level, width):
    return (np.clip((values - (level - width / 2)) / width, 0, 1) * 255).astype(np.uint8)


def annotate(panel, scale, letters, settings, font):
    draw = ImageDraw.Draw(panel)
    length = settings["scale_bar_mm"] * scale
    size = panel.height
    draw.rectangle([8, size - 12, 8 + length, size - 9], fill=(255, 255, 255))
    draw.text((3, size // 2 - 7), letters[0], fill=(255, 220, 0), font=font)
    draw.text((size // 2 - 4, 2), letters[1], fill=(255, 220, 0), font=font)
    return panel


def compose(hu, mask, activation, vmax, spacing, settings, font):
    center = tuple(int(round(value)) for value in np.argwhere(mask).mean(0))
    size, gap = settings["plane_pixels"], settings["gap_pixels"]
    letters = [("R", "A"), ("R", "S"), ("A", "S")]
    rows = []
    for name in ["lung", "mediastinal"]:
        level, width = settings["windows"][name]
        row = []
        for plane, (row_mm, column_mm), marks in zip(planes(hu, center), plane_spacing(spacing), letters):
            panel, scale = fit(windowed(plane, level, width), row_mm, column_mm, size, Image.BILINEAR)
            row.append(annotate(panel.convert("RGB"), scale, marks, settings, font))
        rows.append(row)
    if activation is not None:
        colormap = matplotlib.colormaps["inferno"]
        level, width = settings["windows"]["lung"]
        row = []
        for plane, response, (row_mm, column_mm), marks in zip(planes(hu, center), planes(activation, center), plane_spacing(spacing), letters):
            base, scale = fit(windowed(plane, level, width), row_mm, column_mm, size, Image.BILINEAR)
            strength, _ = fit((np.clip(response / vmax, 0, 1) * 255).astype(np.uint8), row_mm, column_mm, size, Image.BILINEAR)
            fraction = np.asarray(strength, dtype=np.float32) / 255
            colour = colormap(0.35 + 0.65 * fraction)[..., :3] * 255
            alpha = np.where(fraction > 0, settings["overlay_alpha"] * np.clip(fraction * 1.5, 0.25, 1), 0)[..., None]
            blended = np.asarray(base.convert("RGB"), dtype=np.float32) * (1 - alpha) + colour * alpha
            row.append(annotate(Image.fromarray(blended.astype(np.uint8)), scale, marks, settings, font))
        rows.append(row)
    canvas = Image.new("RGB", (3 * size + 2 * gap, len(rows) * size + (len(rows) - 1) * gap), (70, 70, 70))
    for index, row in enumerate(rows):
        for column, panel in enumerate(row):
            canvas.paste(panel, (column * (size + gap), index * (size + gap)))
    return canvas, center


def render(config, root, previous, output):
    torch.set_num_threads(config["render"]["torch_threads"])
    features = {item["key"]: item for item in read_json(output / "features.json")["features"]}
    examples = pd.read_csv(output / "examples.csv", dtype={"PatientID": str})
    examples["source"] = examples.key.map(lambda key: SOURCES[features[key]["site"]])
    needed = examples[["identifier", "source"]].drop_duplicates()
    missing = needed[[not input_path(root, output, row.source, row.identifier).exists() for row in needed.itertuples()]]
    tasks = [(identifier, sorted(group.source), root, output, 1) for identifier, group in missing.groupby("identifier")]
    started = time.monotonic()
    print("INPUTS", len(needed), "needed,", len(tasks), "records to prepare", flush=True)
    with multiprocessing.get_context("spawn").Pool(config["render"]["workers"]) as pool:
        for index, identifier in enumerate(pool.imap_unordered(prepare_inputs, tasks)):
            elapsed = time.monotonic() - started
            print(f"INPUT {index + 1}/{len(tasks)} {identifier} {elapsed:.0f} s, eta {elapsed / (index + 1) * (len(tasks) - index - 1):.0f} s", flush=True)
    font = ImageFont.load_default(size=14)
    settings = config["render"]
    window_mm = {}
    validation_dir = output / "images_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for row in needed.itertuples():
        destination = validation_dir / f"{row.source}_{row.identifier}.png"
        hu, mask, affine = load_volume(input_path(root, output, row.source, row.identifier))
        spacing = nib.affines.voxel_sizes(affine)
        window_mm[row.source] = float(round(max(np.asarray(hu.shape) * spacing)))
        if not destination.exists():
            compose(hu, mask, None, None, spacing, settings, font)[0].save(destination)
    records = []
    for key, group in examples[examples.stage == "generation"].groupby("key", sort=False):
        feature = features[key]
        source = SOURCES[feature["site"]]
        maps = {}
        for row in group.itertuples():
            hu, mask, affine = load_volume(input_path(root, output, source, row.identifier))
            maps[row.example_id] = (hu, mask, affine, *feature_map(root, previous, feature, row.identifier, affine, hu.shape))
        vmax = max(float(item[3].max()) for item in maps.values())
        assert vmax > 0, key
        destination = output / "images_generation" / key
        destination.mkdir(parents=True, exist_ok=True)
        for row in group.itertuples():
            hu, mask, affine, activation, case_value = maps[row.example_id]
            if feature["role"] != "shuffled":
                assert abs(case_value - row.value) <= 1e-3 * max(1.0, row.value), (key, row.identifier, case_value, row.value)
            image, center = compose(hu, mask, activation, vmax, nib.affines.voxel_sizes(affine), settings, font)
            image.save(destination / f"{row.example_id}.png")
            in_planes = [float(activation[center[0], :, :].max()), float(activation[:, center[1], :].max()), float(activation[:, :, center[2]].max())]
            records.append({"key": key, "example_id": row.example_id, "identifier": row.identifier, "case_value": case_value, "volume_max": float(activation.max()), "plane_max": max(in_planes), "vmax": vmax})
        print("RENDERED", key, len(group), flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "render_checks.csv", index=False)
    write_json(output / "render.json", {"window_mm": window_mm, "generation_images": len(frame), "validation_images": len(needed),
                                         "plane_max_over_volume_max_median": float((frame.plane_max / frame.volume_max.where(frame.volume_max > 0)).median()), "created_at": timestamp()})
    print("RENDER done", len(frame), "generation images", len(needed), "validation images", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["select", "render", "measure"], required=True)
    args = parser.parse_args()
    config, root, previous, candidates, output = paths()
    if args.stage == "select":
        select(config, root, candidates, output)
    elif args.stage == "render":
        render(config, root, previous, output)
    else:
        measure(config, root, candidates, output)


if __name__ == "__main__":
    main()

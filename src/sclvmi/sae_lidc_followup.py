import argparse
import io
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage, stats
from sklearn.linear_model import Lasso, lasso_path
from sklearn.model_selection import KFold

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .models import load_encoder
from .sae import SpatialDictionary
from .sae_concepts import dictionaries, members as profile_members, stable_mask
from .sae_context import RecordDataset, configuration as campaign_configuration, unchanged, vista_sites
from .sae_encoders import fmcib_prefix, parameter_digest, random_encoder
from .sae_interp_candidates import residual_ranks
from .sae_lidc import configuration as lidc_configuration, directories as lidc_directories, region_statistics, site_associations


CONFIG_PATH = ROOT / "configs/sae_lidc_followup.json"
# trained, random-weight, distance and valid-position keys of each site's case arrays
KEYS = {"fmcib:layer2": ("layer2", "layer2_random", "layer2_distance", None), "vista:stage2": ("stage2", "stage2_random", "stage2_distance", None),
        "fmcib:layer1_context": ("context_trained", "context_random", "context_distance", "context_valid"), "vista:stage3": ("stage3", "stage3_random", "stage3_distance", None)}
GRID_AFFINE = {"fmcib:layer2": "layer2_grid_affine_ras", "vista:stage2": "stage2_grid_affine_ras"}
SHORT = {"subtlety": "sub", "sphericity": "sph", "margin": "mar", "lobulation": "lob", "spiculation": "spi", "texture": "tex", "malignancy": "mal", "calcification": "cal"}
# reader question for each rating, and whether a higher rating means the set named by the question (margin: a higher rating is a sharper margin)
QUESTIONS = {"spiculation": ("which set is more spiculated", True), "lobulation": ("which set is more lobulated", True), "margin": ("which set has less sharply defined margins", False),
             "texture": ("which set is more solid", True), "calcification": ("which set is more calcified", True), "malignancy": ("which set looks more suspicious for malignancy", True),
             "subtlety": ("which set is more conspicuous", True), "contact": ("which set more often touches the pleura, chest wall or mediastinum", True)}


def configuration():
    return read_json(CONFIG_PATH)


def directories():
    config, storage, campaign = configuration(), context()[0], campaign_configuration()
    _, crops, source, campaign_root = lidc_directories()
    return crops, source, Path(storage["runs"]) / config["run_id"], campaign_root, Path(storage["activations"]) / campaign["previous_suite"]


def lidc_case(name, identifier):
    crops, source, run, _, _ = directories()
    model = name.split(":")[0]
    trained, randomized, distance, valid = KEYS[name]
    base = source if name in GRID_AFFINE else run
    with np.load(base / "cases" / model / f"{identifier}.npz") as saved:
        return saved[trained], saved[randomized], saved[distance], saved[valid] if valid else None


def luna25_case(name, identifier):
    _, _, _, campaign_root, previous = directories()
    model, site = name.split(":")
    trained, randomized, distance, valid = KEYS[name]
    with np.load(io.BytesIO((campaign_root / "cases" / model / f"{identifier}.npz").read_bytes())) as new:
        if name == "fmcib:layer1_context":
            return new[trained], new[randomized], new[distance], new[valid]
        with np.load(io.BytesIO((previous / model / "cases" / f"{identifier}.npz").read_bytes())) as old:
            return old[site], new[randomized], new[distance], None


def tokens(values, distance, valid, device):
    keep = distance.reshape(-1) <= 10
    if valid is not None:
        keep &= valid.reshape(-1)
    keep = np.flatnonzero(keep)
    return torch.from_numpy(values.reshape(values.shape[0], -1).T[keep]).to(device), torch.from_numpy(distance.reshape(-1)[keep]).to(device)


def encode(device, verify):
    campaign = campaign_configuration()
    crops, _, run, campaign_root, previous = directories()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoders = {}
    for name in ["fmcib", "vista"]:
        randomized = random_encoder(name, campaign["random_encoder"]["initialization_seed"])
        assert parameter_digest(randomized) == campaign["whole_ct"]["random_encoder_sha256"][name], name
        encoders[name] = (load_encoder(name)[0].to(device).eval(), randomized.to(device))
    first, count = campaign["context"]["first_position"], campaign["context"]["positions"]
    if verify:
        frame = pd.read_csv(campaign_root / "members.csv", dtype={"PatientID": str}).sort_values("identifier").iloc[:: 97][:verify]
        destination = run / "verification"
    else:
        frame = pd.read_csv(crops / "members.csv", dtype={"PatientID": str})
        frame = frame.assign(image=frame["image"].map(lambda value: str(crops / value)), mask=frame["mask"].map(lambda value: str(crops / value)), label=-1)
        destination = run / "cases"
    for model in encoders:
        (destination / model).mkdir(parents=True, exist_ok=True)
    pending = frame[[not (destination / "vista" / f"{value}.done").exists() for value in frame.identifier]]
    print(f"ENCODE {len(pending)} of {len(frame)} records pending ({'LUNA25 verification' if verify else 'LIDC'})", flush=True)
    loader = torch.utils.data.DataLoader(RecordDataset(pending, campaign, []), batch_size=None, collate_fn=unchanged, num_workers=6, persistent_workers=False, prefetch_factor=4)
    started, checks = time.monotonic(), []
    window = slice(first, first + count)
    for number, item in enumerate(loader, 1):
        identifier = item["identifier"]
        with torch.inference_mode():
            cube = torch.from_numpy(item["cube"])[None, None].to(device)
            vista = torch.from_numpy(item["vista"])[None, None].to(device)
            values = {
                "fmcib": {"context_trained": fmcib_prefix(encoders["fmcib"][0], cube, "layer1")[0, :, window, window, window].cpu().numpy(),
                          "context_random": fmcib_prefix(encoders["fmcib"][1], cube, "layer1")[0, :, window, window, window].cpu().numpy(),
                          "context_valid": item["arrays"]["fmcib"]["context_valid"], "context_distance": item["arrays"]["fmcib"]["context_distance"]},
                "vista": {"stage3": vista_sites(encoders["vista"][0], vista)["stage3"][0].cpu().numpy(), "stage3_random": vista_sites(encoders["vista"][1], vista)["stage3"][0].cpu().numpy(),
                          "stage3_distance": item["arrays"]["vista"]["stage3_distance"]},
            }
        if verify:
            with np.load(io.BytesIO((campaign_root / "cases/fmcib" / f"{identifier}.npz").read_bytes())) as new:
                checks.append({"identifier": identifier, "site": "fmcib:layer1_context", **{key: float(np.abs(values["fmcib"][key].astype(np.float64) - new[key].astype(np.float64)).max()) for key in ["context_trained", "context_random", "context_distance"]},
                               "valid_equal": bool((values["fmcib"]["context_valid"] == new["context_valid"]).all())})
            with np.load(io.BytesIO((campaign_root / "cases/vista" / f"{identifier}.npz").read_bytes())) as new, np.load(io.BytesIO((previous / "vista/cases" / f"{identifier}.npz").read_bytes())) as old:
                checks.append({"identifier": identifier, "site": "vista:stage3", "stage3": float(np.abs(values["vista"]["stage3"] - old["stage3"]).max()),
                               "stage3_random": float(np.abs(values["vista"]["stage3_random"] - new["stage3_random"]).max()), "stage3_distance": float(np.abs(values["vista"]["stage3_distance"] - new["stage3_distance"]).max())})
        else:
            for model, arrays in values.items():
                path = destination / model / f"{identifier}.npz"
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **arrays)
                temporary.replace(path)
        (destination / "vista" / f"{identifier}.done").write_text(timestamp(), encoding="utf-8")
        if number % 50 == 0 or number == len(pending):
            rate = number / (time.monotonic() - started)
            print(f"ENCODE {number}/{len(pending)} records, {rate:.2f} records/s, eta {(len(pending) - number) / rate / 60:.1f} min", flush=True)
    if verify:
        table = pd.DataFrame(checks)
        table.to_csv(destination / "luna25_checks.csv", index=False)
        maximum = table.drop(columns=["identifier", "site"]).max(numeric_only=True).to_dict()
        write_json(destination / "luna25_checks.json", {"records": int(table.identifier.nunique()), "maximum": maximum, "valid_equal": bool(table.valid_equal.dropna().all()), "created_at": timestamp()})
        print(maximum, flush=True)
        assert table.valid_equal.dropna().all() and all(value < 1e-3 for value in maximum.values()), maximum


@torch.inference_mode()
def peaks(device):
    config, campaign = configuration(), campaign_configuration()
    _, _, run, campaign_root, _ = directories()
    train = profile_members(campaign_root).query("split == 'train'")
    statistics = lidc_configuration()["activation"]["statistics"]
    run.mkdir(parents=True, exist_ok=True)
    for name in config["extra_sites"]:
        model, site = name.split(":")
        destination = run / f"peaks_{model}_{site}.csv"
        if destination.exists():
            continue
        trained, randomized, _ = dictionaries(campaign_root, model, site, config["reference_seed"])
        saes, features = {}, {}
        for encoder, row in [("trained", trained), ("random", randomized)]:
            features[encoder] = torch.from_numpy(np.flatnonzero(stable_mask(campaign_root, row["stability_group"], campaign)[0])).to(device)
            saes[encoder] = SpatialDictionary(campaign_root / "runs" / row["run_id"] / "dictionary.pt", device=device)
        totals = {encoder: [] for encoder in saes}
        started = time.monotonic()
        for number, identifier in enumerate(train.identifier, 1):
            trained_values, random_values, distance, valid = luna25_case(name, identifier)
            for encoder, values in [("trained", trained_values), ("random", random_values)]:
                selected, near = tokens(values, distance, valid, device)
                totals[encoder].append(region_statistics(saes[encoder].encode(selected)[:, features[encoder]], near)[:, :2].cpu().numpy())
            if number % 200 == 0 or number == len(train):
                print(f"PEAKS {name} {number}/{len(train)} {time.monotonic() - started:.0f} s", flush=True)
        rows = []
        for encoder in saes:
            means = np.nanmean(np.stack(totals[encoder]), 0)
            for feature, values in zip(features[encoder].cpu().numpy(), means):
                rows.append({"site": name, "encoder": encoder, "feature": int(feature), "within_4mm_mean": values[0], "near_4_10mm_mean": values[1], "peak_region": statistics[int(np.nanargmax(values))],
                             "patients_with_region": int(np.isfinite(np.stack(totals[encoder])[:, 0, 0]).sum())})
        pd.DataFrame(rows).to_csv(destination, index=False)


@torch.inference_mode()
def activations(device):
    config, campaign = configuration(), campaign_configuration()
    crops, _, run, campaign_root, _ = directories()
    frame = pd.read_csv(crops / "members.csv", dtype={"PatientID": str})
    for name in config["extra_sites"]:
        model, site = name.split(":")
        destination = run / f"activations_{model}_{site}.npz"
        if destination.exists():
            continue
        trained, randomized, others = dictionaries(campaign_root, model, site, config["reference_seed"])
        stable, matched, seeds = stable_mask(campaign_root, trained["stability_group"], campaign)
        assert seeds[0] == config["reference_seed"]
        load = lambda row: SpatialDictionary(campaign_root / "runs" / row["run_id"] / "dictionary.pt", device=device)
        trained_sae, random_sae = load(trained), load(randomized)
        stable_index = np.flatnonzero(stable)
        other_saes = {int(seed): (load(others[int(seed)]), torch.from_numpy(matched[0, position, stable_index].astype(np.int64)).to(device)) for position, seed in enumerate(seeds) if position > 0}
        result = {"trained": [], "random": [], "matched": {seed: [] for seed in other_saes}}
        for identifier in frame.identifier:
            trained_values, random_values, distance, valid = lidc_case(name, identifier)
            selected, near = tokens(trained_values, distance, valid, device)
            random_selected, _ = tokens(random_values, distance, valid, device)
            result["trained"].append(region_statistics(trained_sae.encode(selected), near).cpu().numpy())
            result["random"].append(region_statistics(random_sae.encode(random_selected), near).cpu().numpy())
            for seed, (sae, columns) in other_saes.items():
                result["matched"][seed].append(region_statistics(sae.encode(selected)[:, columns], near).cpu().numpy())
        np.savez_compressed(destination, identifiers=frame.identifier.to_numpy(dtype=str), trained=np.stack(result["trained"]).astype(np.float32), random=np.stack(result["random"]).astype(np.float32),
                            matched=np.stack([np.stack(result["matched"][seed]) for seed in sorted(result["matched"])]).astype(np.float32), matched_seeds=np.asarray(sorted(result["matched"])),
                            stable=stable, random_stable=stable_mask(campaign_root, randomized["stability_group"], campaign)[0])
        print("ACTIVATIONS", name, len(frame), "nodules", int(stable.sum()), "stable features", flush=True)
    for name in config["all_sites"]:
        model, site = name.split(":")
        destination = run / f"raw_{model}_{site}.npz"
        if destination.exists():
            continue
        result = {"trained": [], "random": []}
        for identifier in frame.identifier:
            trained_values, random_values, distance, valid = lidc_case(name, identifier)
            for encoder, values in [("trained", trained_values), ("random", random_values)]:
                selected, near = tokens(values, distance, valid, device)
                result[encoder].append(region_statistics(selected, near)[:, :2].cpu().numpy())
        np.savez_compressed(destination, identifiers=frame.identifier.to_numpy(dtype=str), trained=np.stack(result["trained"]).astype(np.float32), random=np.stack(result["random"]).astype(np.float32))
        print("RAW", name, flush=True)


def probe(inputs, values_ratings, size, halves, settings, generator):
    discovery, confirmation = halves["discovery"], halves["confirmation"]
    finite = np.isfinite(inputs).all(0)
    left, right = residual_ranks(inputs[discovery][:, finite], size[discovery]), residual_ranks(inputs[confirmation][:, finite], size[confirmation])
    scale = left.std(0)
    keep = scale > 1e-9
    left, right = left[:, keep] / scale[keep], right[:, keep] / scale[keep]
    rows = []
    folds = list(KFold(5, shuffle=True, random_state=settings["seed"]).split(left))
    for index in range(values_ratings.shape[1]):
        target, held = residual_ranks(values_ratings[discovery][:, [index]], size[discovery])[:, 0], residual_ranks(values_ratings[confirmation][:, [index]], size[confirmation])[:, 0]
        target_scale = target.std()
        target, held = target / target_scale, held / target_scale
        largest = np.abs(left.T @ target).max() / len(target)
        alphas = np.geomspace(largest, largest * 1e-3, 50)
        errors = []
        for train, test in folds:
            _, coefficients, _ = lasso_path(left[train], target[train] - target[train].mean(), alphas=alphas, max_iter=20000)
            errors.append(np.square(target[test][:, None] - target[train].mean() - left[test] @ coefficients).mean(0))
        errors = np.asarray(errors)
        mean, error = errors.mean(0), errors.std(0, ddof=1) / np.sqrt(len(folds))
        best = int(np.argmin(mean))
        chosen = alphas[np.flatnonzero(mean <= mean[best] + error[best])[0]]
        model = Lasso(alpha=chosen, max_iter=20000).fit(left, target)
        prediction = right @ model.coef_
        nonzero = int((model.coef_ != 0).sum())
        if nonzero == 0:
            rows.append({"score": 0.0, "low": 0.0, "high": 0.0, "nonzero": 0})
            continue
        score = float(np.corrcoef(prediction, held)[0, 1])
        draws = []
        for _ in range(settings["bootstrap"]):
            sample = generator.integers(len(held), size=len(held))
            draws.append(np.corrcoef(prediction[sample], held[sample])[0, 1])
        rows.append({"score": score, "low": float(np.nanpercentile(draws, 2.5)), "high": float(np.nanpercentile(draws, 97.5)), "nonzero": nonzero})
    return rows


def analyze():
    config, lidc = configuration(), lidc_configuration()
    crops, source, run, _, _ = directories()
    members_table = pd.read_csv(crops / "members.csv", dtype={"PatientID": str})
    ratings = lidc["ratings"]["ordinal"] + ["calcification"]
    statistics = lidc["activation"]["statistics"]
    destination = run / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(lidc["association"]["bootstrap_seed"])
    tables, summaries = [], {}
    for name in config["extra_sites"]:
        model, site = name.split(":")
        with np.load(run / f"activations_{model}_{site}.npz") as saved:
            assert (saved["identifiers"] == members_table.identifier.to_numpy()).all()
            arrays = {key: saved[key] for key in saved.files}
        peak_table = pd.read_csv(run / f"peaks_{model}_{site}.csv")
        peaks = {encoder: rows.set_index("feature").peak_region.map(statistics.index) for encoder, rows in peak_table.groupby("encoder")}
        result = site_associations(name, arrays, members_table, ratings, peaks, None, lidc, generator)
        tables.append(result["table"])
        summaries[name] = result["summary"]
    extra = pd.concat(tables, ignore_index=True)
    extra.to_csv(destination / "associations_extra_sites.csv", index=False)
    associations = pd.concat([pd.read_csv(source / "analysis/associations.csv"), extra], ignore_index=True)
    agreement = pd.read_csv(source / "analysis/ceiling.csv").set_index("rating").reader_vs_others_size_adjusted_rho
    resolution = []
    for (name, rating), group in associations.groupby(["site", "rating"], sort=False):
        row = {"site": name, "rating": rating, "radiologist_agreement": agreement[rating]}
        for encoder in ["trained", "random"]:
            confirmed = group[(group.encoder == encoder) & group.active & group.confirmed]
            row[f"{encoder}_analysed"] = int(((group.encoder == encoder) & group.active).sum())
            row[f"{encoder}_confirmed"] = len(confirmed)
            row[f"{encoder}_largest"] = confirmed.confirmation_rho.abs().max() if len(confirmed) else np.nan
        replicated = group[(group.encoder == "trained") & group.confirmed & (group.replicated_seeds >= 6)]
        row["trained_replicated"] = len(replicated)
        row["trained_replicated_largest"] = replicated.confirmation_rho.abs().max() if len(replicated) else np.nan
        resolution.append(row)
    resolution = pd.DataFrame(resolution)
    resolution.to_csv(destination / "resolution.csv", index=False)
    probes = []
    settings = config["probe"]
    probe_generator = np.random.default_rng(settings["seed"])
    for name in config["all_sites"]:
        model, site = name.split(":")
        activation_path = (source if name in GRID_AFFINE else run) / f"activations_{model}_{site}.npz"
        with np.load(activation_path) as saved, np.load(run / f"raw_{model}_{site}.npz") as raw:
            inputs = {"dictionary": saved["trained"][:, saved["stable"], :2], "random_dictionary": saved["random"][:, saved["random_stable"], :2], "raw": raw["trained"], "random_raw": raw["random"]}
        valid = np.isfinite(inputs["dictionary"]).all((1, 2))
        frame = members_table[valid].reset_index(drop=True)
        halves = {half: (frame.split == half).to_numpy() for half in ["discovery", "confirmation"]}
        size = np.log(frame.consensus_volume_mm3.to_numpy())[:, None]
        values_ratings = frame[ratings].to_numpy(dtype=float)
        for kind, values in inputs.items():
            values = values[valid].reshape(int(valid.sum()), -1)
            for rating, row in zip(ratings, probe(values, values_ratings, size, halves, settings, probe_generator)):
                probes.append({"site": name, "input": kind, "rating": rating, "columns": values.shape[1], "nodules": int(valid.sum()), **row})
            print("PROBE", name, kind, flush=True)
    probes = pd.DataFrame(probes)
    probes.to_csv(destination / "probe.csv", index=False)
    write_json(destination / "summary.json", {"extra_sites": summaries, "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})
    resolution_figure(resolution, ratings, destination)
    probe_figure(probes, resolution, ratings, destination)
    print(resolution.round(3).to_string(index=False), flush=True)
    print(probes.pivot_table(index=["site", "rating"], columns="input", values="score").round(3).to_string(), flush=True)


def style():
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    return ink, muted, surface, grid


def finish(fig, axes, surface, grid, destination, stem):
    for axis in np.ravel(axes):
        axis.set_facecolor(surface)
        axis.grid(axis="y", color=grid, linewidth=0.6)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"{stem}.{suffix}", dpi=170, facecolor=surface)
    plt.close(fig)


def resolution_figure(resolution, ratings, destination):
    ink, muted, surface, grid = style()
    colors = {"fmcib:layer1_context": "#8fb8ea", "fmcib:layer2": "#2a78d6", "vista:stage2": "#0e8a6a", "vista:stage3": "#8fd1bd"}
    labels = {"fmcib:layer1_context": "FMCIB layer1 context (4 mm grid)", "fmcib:layer2": "FMCIB layer2 (7 mm grid)", "vista:stage2": "VISTA3D stage2 (6 mm grid)", "vista:stage3": "VISTA3D stage3 (12 mm grid)"}
    fig, axis = plt.subplots(figsize=(10, 4.2), facecolor=surface)
    offsets = dict(zip(labels, [-0.27, -0.09, 0.09, 0.27]))
    for name, offset in offsets.items():
        rows = resolution[resolution.site == name].set_index("rating").loc[ratings]
        axis.bar(np.arange(len(ratings)) + offset, rows.trained_replicated_largest.fillna(0), width=0.17, color=colors[name], label=labels[name])
    agreement = resolution.drop_duplicates("rating").set_index("rating").loc[ratings].radiologist_agreement
    axis.hlines(agreement, np.arange(len(ratings)) - 0.4, np.arange(len(ratings)) + 0.4, color=ink, linewidth=2, label="one radiologist vs the others (size-adjusted)")
    axis.set_xticks(np.arange(len(ratings)), ratings)
    axis.set_ylabel("largest confirmed, seed-replicated\n|size-adjusted partial Spearman|")
    fig.legend(*axis.get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    finish(fig, axis, surface, grid, destination, "resolution")


def probe_figure(probes, resolution, ratings, destination):
    ink, muted, surface, grid = style()
    colors = {"dictionary": "#2a78d6", "raw": "#0b0b0b", "random_dictionary": "#eb6834", "random_raw": "#b9b8b3"}
    labels = {"dictionary": "SAE dictionary (few features)", "raw": "encoder channels", "random_dictionary": "random-weight dictionary", "random_raw": "random-weight encoder channels"}
    sites = list(dict.fromkeys(probes.site))
    fig, axes = plt.subplots(len(sites), 1, figsize=(10, 2.9 * len(sites)), facecolor=surface, squeeze=False)
    for axis, name in zip(axes[:, 0], sites):
        for offset, kind in zip([-0.27, -0.09, 0.09, 0.27], labels):
            rows = probes[(probes.site == name) & (probes.input == kind)].set_index("rating").loc[ratings]
            axis.errorbar(np.arange(len(ratings)) + offset, rows.score, yerr=[rows.score - rows.low, rows.high - rows.score], fmt="o", markersize=4, color=colors[kind], elinewidth=1, label=labels[kind])
        single = resolution[resolution.site == name].set_index("rating").loc[ratings].trained_largest
        axis.scatter(np.arange(len(ratings)), single.fillna(0), marker="D", s=22, facecolor="none", edgecolor="#7a4fd1", label="best single feature", zorder=3)
        agreement = resolution.drop_duplicates("rating").set_index("rating").loc[ratings].radiologist_agreement
        axis.hlines(agreement, np.arange(len(ratings)) - 0.4, np.arange(len(ratings)) + 0.4, color=ink, linewidth=1.5, label="one radiologist vs the others")
        axis.axhline(0, color=muted, linewidth=0.8)
        axis.set_xticks(np.arange(len(ratings)), ratings)
        axis.set_title(name, loc="left")
        axis.set_ylabel("confirmation correlation")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    finish(fig, axes, surface, grid, destination, "probe")


def size_matched_low(pool, target, high, count, neighbours=10):
    # For each high nodule, the nodule with the lowest target among the unused nodules nearest in log volume.
    candidates = pool[~pool.identifier.isin(high.identifier)].assign(target=target, log_volume=np.log(pool.volume))
    chosen = []
    for volume in np.log(high.volume.to_numpy())[:count]:
        nearest = (candidates.log_volume - volume).abs().nsmallest(neighbours).index
        index = candidates.loc[nearest, "target"].idxmin()
        chosen.append(index)
        candidates = candidates.drop(index)
    return pool.loc[chosen]


def feature_map(name, sae, feature, values, geometry):
    codes = sae.encode(torch.from_numpy(values.reshape(values.shape[0], -1).T).to(sae.mean.device))[:, feature]
    return codes.reshape(values.shape[1:]).cpu().numpy(), np.asarray(geometry[GRID_AFFINE[name]])


def slice_view(image_path, mask_path, field_mm, plane):
    image, mask = nib.load(image_path), nib.load(mask_path)
    values, labels = np.asarray(image.dataobj, dtype=np.float32), np.asarray(mask.dataobj) > 0
    spacing = nib.affines.voxel_sizes(image.affine)
    centre = np.asarray(values.shape) // 2
    half = np.round(field_mm / 2 / spacing).astype(int)
    lows, highs = np.maximum(centre - half, 0), np.minimum(centre + half, values.shape)
    if plane == "axial":
        index = np.mgrid[lows[0]:highs[0], lows[1]:highs[1]]
        voxels = np.stack([index[0], index[1], np.full(index[0].shape, centre[2])])
        picture, contour = values[lows[0]:highs[0], lows[1]:highs[1], centre[2]].T, labels[lows[0]:highs[0], lows[1]:highs[1], centre[2]].T
        extent = (0, (highs[0] - lows[0]) * spacing[0], (highs[1] - lows[1]) * spacing[1], 0)
        point = ((centre[0] - lows[0] + 0.5) * spacing[0], (centre[1] - lows[1] + 0.5) * spacing[1])
    else:
        index = np.mgrid[lows[0]:highs[0], lows[2]:highs[2]]
        voxels = np.stack([index[0], np.full(index[0].shape, centre[1]), index[1]])
        picture, contour = values[lows[0]:highs[0], centre[1], lows[2]:highs[2]].T[::-1], labels[lows[0]:highs[0], centre[1], lows[2]:highs[2]].T[::-1]
        extent = (0, (highs[0] - lows[0]) * spacing[0], (highs[2] - lows[2]) * spacing[2], 0)
        point = ((centre[0] - lows[0] + 0.5) * spacing[0], (highs[2] - centre[2] - 0.5) * spacing[2])
    return picture, contour, extent, voxels, image.affine, point


def overlay(code_volume, grid_affine, voxels, affine, plane):
    world = nib.affines.apply_affine(affine, voxels.reshape(3, -1).T)
    grid = nib.affines.apply_affine(np.linalg.inv(grid_affine), world)
    sampled = ndimage.map_coordinates(code_volume, grid.T, order=1, mode="constant", cval=0.0).reshape(voxels.shape[1:]).T
    return sampled if plane == "axial" else sampled[::-1]


def matched_sets(pool, target, count):
    order = np.argsort(-target, kind="stable")
    high = pool.iloc[order[:count]]
    return high, size_matched_low(pool, pd.Series(target, index=pool.index), high, count)


def pages(device):
    config, lidc, campaign = configuration(), lidc_configuration(), campaign_configuration()
    crops, source, run, campaign_root, previous = directories()
    settings = config["pages"]
    destination = run / "pages"
    destination.mkdir(parents=True, exist_ok=True)
    statistics = lidc["activation"]["statistics"]
    clear = pd.read_csv(source / "analysis/clear_features.csv")
    explanations = pd.read_csv(Path(context()[0]["runs"]) / lidc["radiomics_run"] / "analysis/explanations.csv")
    members_table = pd.read_csv(crops / "members.csv", dtype={"PatientID": str}).assign(volume=lambda frame: frame.consensus_volume_mm3)
    luna = profile_members(campaign_root).query("split == 'train' and identifier != '119347_2_20000102'")
    references = pd.read_csv(campaign_root / "nodule_references.csv")[["identifier", "volume_mm3"]]
    luna = luna.merge(references, on="identifier").rename(columns={"volume_mm3": "volume"})
    ink, muted, surface, _ = style()
    rows = []
    for name, features in clear.groupby("site", sort=False).feature:
        model, site = name.split(":")
        trained, _, _ = dictionaries(campaign_root, model, site, lidc["reference_seed"])
        sae = SpatialDictionary(campaign_root / "runs" / trained["run_id"] / "dictionary.pt", device=device)
        with np.load(source / f"activations_{model}_{site}.npz") as saved:
            lidc_values = saved["trained"]
        with np.load(Path(context()[0]["runs"]) / lidc["radiomics_run"] / f"activations_{model}_{site}.npz") as saved:
            luna_identifiers, luna_values = saved["identifiers"], saved["trained"]
        for feature in features.unique():
            region = statistics.index(explanations[(explanations.site == name) & (explanations.encoder == "trained") & (explanations.feature == feature)].peak_region.iloc[0])
            confirmed = clear[(clear.site == name) & (clear.feature == feature)]
            discovery = members_table[members_table.split == "discovery"]
            lidc_target = lidc_values[discovery.index, feature, region]
            luna_frame = luna.set_index("identifier").loc[luna_identifiers[np.isin(luna_identifiers, luna.identifier)]].reset_index()
            luna_target = luna_values[np.isin(luna_identifiers, luna.identifier), feature, region]
            record = {"site": name, "feature": int(feature), "region": statistics[region], "confirmed": "; ".join(f"{row.rating} {row.confirmation_rho:+.2f}" for row in confirmed.itertuples())}
            high30, low30 = matched_sets(discovery.reset_index(drop=True), lidc_target, 30)
            for row in confirmed.itertuples():
                result = stats.mannwhitneyu(high30[row.rating], low30[row.rating])
                record[f"{row.rating}_high"], record[f"{row.rating}_low"], record[f"{row.rating}_p"] = high30[row.rating].mean(), low30[row.rating].mean(), result.pvalue
                record[f"{row.rating}_agrees"] = bool(np.sign(high30[row.rating].mean() - low30[row.rating].mean()) == np.sign(row.confirmation_rho))
            luna_high30, luna_low30 = matched_sets(luna_frame, luna_target, 30)
            record["luna25_malignant_high"], record["luna25_malignant_low"] = luna_high30.label.mean(), luna_low30.label.mean()
            record["lidc_volume_high"], record["lidc_volume_low"] = high30.volume.median(), low30.volume.median()
            rows.append(record)
            panels = []
            for dataset, frame, target in [("LIDC", discovery.reset_index(drop=True), lidc_target), ("LUNA25", luna_frame, luna_target)]:
                high, low = matched_sets(frame, target, settings["high"])
                for group, chosen in [("high", high), ("low", low)]:
                    for item in chosen.itertuples():
                        if dataset == "LIDC":
                            with np.load(source / "cases" / model / f"{item.identifier}.npz") as saved:
                                values = saved[KEYS[name][0]]
                            geometry = read_json(source / "cases" / model / f"{item.identifier}.json")
                            image, mask = crops / item.image, crops / item.mask
                            caption = " ".join(f"{SHORT[rating]} {getattr(item, rating):g}" for rating in ["spiculation", "lobulation", "margin", "texture"]) + "\n" + " ".join(f"{SHORT[rating]} {getattr(item, rating):.2g}" for rating in ["malignancy", "calcification", "subtlety"])
                        else:
                            with np.load(io.BytesIO((previous / model / "cases" / f"{item.identifier}.npz").read_bytes())) as saved:
                                values = saved[site]
                            geometry = read_json(campaign_root / "cases" / model / f"{item.identifier}.json")
                            image, mask = Path(item.image), Path(item.mask)
                            caption = f"label {int(item.label)}"
                        code_volume, grid_affine = feature_map(name, sae, feature, values, geometry)
                        picture, contour, extent, voxels, affine, _ = slice_view(image, mask, 50.0, "axial")
                        panels.append((dataset, group, item.identifier, picture, contour, extent, overlay(code_volume, grid_affine, voxels, affine, "axial"), f"{caption} | {item.volume:.0f} mm3"))
            limit = max(float(np.percentile(np.concatenate([panel[6].ravel() for panel in panels if panel[1] == "high"]), 99)), 1e-6)
            fig, axes = plt.subplots(4, settings["high"], figsize=(1.55 * settings["high"], 7.6), facecolor=surface)
            for axis, (dataset, group, identifier, picture, contour, extent, heat, caption) in zip(axes.ravel(), panels):
                axis.imshow(picture, cmap="gray", vmin=-1000, vmax=400, extent=extent)
                axis.imshow(np.ma.masked_less(heat, 0.05 * limit), cmap="inferno", vmin=0, vmax=limit, alpha=0.45, extent=extent)
                axis.contour(np.linspace(extent[0], extent[1], contour.shape[1]), np.linspace(extent[3], extent[2], contour.shape[0]), contour.astype(float), levels=[0.5], colors="#3fd0ff", linewidths=0.6)
                axis.set_title(caption, fontsize=4.6, color=ink)
                axis.set_xticks([])
                axis.set_yticks([])
            for row_index, label in enumerate(["LIDC discovery: highest", "LIDC: size-matched low", "LUNA25 training: highest", "LUNA25: size-matched low"]):
                axes[row_index, 0].set_ylabel(label, fontsize=7)
            fig.suptitle(f"{name} feature {feature} ({statistics[region]}); confirmed on LIDC: {record['confirmed']}", fontsize=9, x=0.01, ha="left")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(destination / f"{model}_{site}_{feature}.png", dpi=150, facecolor=surface)
            plt.close(fig)
            print("PAGE", name, feature, flush=True)
    pd.DataFrame(rows).to_csv(destination / "consistency.csv", index=False)


def task(device):
    config, lidc = configuration(), lidc_configuration()
    crops, source, run, campaign_root, _ = directories()
    settings = config["task"]
    review = pd.read_csv(ROOT / settings["reader_review"])
    kept = review[review.keep]
    statistics = lidc["activation"]["statistics"]
    explanations = pd.read_csv(Path(context()[0]["runs"]) / lidc["radiomics_run"] / "analysis/explanations.csv")
    clear = pd.read_csv(source / "analysis/clear_features.csv")
    prespecified = pd.read_csv(source / "analysis/prespecified.csv")
    members_table = pd.read_csv(crops / "members.csv", dtype={"PatientID": str}).assign(volume=lambda frame: frame.consensus_volume_mm3)
    pool = members_table[members_table.split == "confirmation"].reset_index(drop=True)
    generator = np.random.default_rng(config["probe"]["seed"])
    destination = run / "task"
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for row in kept.itertuples():
        model, site = row.site.split(":")
        region = statistics.index(explanations[(explanations.site == row.site) & (explanations.encoder == "trained") & (explanations.feature == row.feature)].peak_region.iloc[0])
        with np.load(source / f"activations_{model}_{site}.npz") as saved:
            target = saved["trained"][members_table.index[members_table.split == "confirmation"], row.feature, region]
        high, low = matched_sets(pool, target, settings["set_size"])
        predicted = {rating: ("H" if rho > 0 else "L") for rating, rho in clear[(clear.site == row.site) & (clear.feature == row.feature)][["rating", "confirmation_rho"]].itertuples(index=False)}
        contact = prespecified[(prespecified.site == row.site) & (prespecified.feature == row.feature) & (prespecified.measure == "lung_boundary_contact_fraction") & (prespecified.p_holm < 0.05)]
        if len(contact):
            predicted["contact"] = "H" if contact.rho.iloc[0] > 0 else "L"
        items.append({"kind": "feature", "site": row.site, "feature": int(row.feature), "H": high, "L": low, "predicted": predicted})
    for _ in range(settings["controls"]):
        first = pool.iloc[generator.choice(len(pool), settings["set_size"], replace=False)]
        second = size_matched_low(pool.assign(volume=pool.volume), pd.Series(np.ones(len(pool)), index=pool.index), first, settings["set_size"])
        items.append({"kind": "control", "site": None, "feature": None, "H": first, "L": second, "predicted": {}})
    order = generator.permutation(len(items))
    ink, muted, surface, _ = style()
    key = []
    for number, position in enumerate(order, 1):
        item = items[position]
        high_first = bool(generator.integers(2))
        sets = {"A": item["H"] if high_first else item["L"], "B": item["L"] if high_first else item["H"]}
        fig, axes = plt.subplots(4, settings["set_size"], figsize=(1.6 * settings["set_size"], 7.2), facecolor=surface)
        for block, label in enumerate(["A", "B"]):
            for column, nodule in enumerate(sets[label].itertuples()):
                for offset, plane in enumerate(["axial", "coronal"]):
                    picture, _, extent, _, _, (x, y) = slice_view(crops / nodule.image, crops / nodule.mask, 50.0, plane)
                    axis = axes[2 * block + offset, column]
                    axis.imshow(picture, cmap="gray", vmin=-1000, vmax=400, extent=extent)
                    # Edge ticks point at the target nodule without covering it.
                    axis.plot([0, 4], [y, y], color="#ffd400", linewidth=1.5)
                    axis.plot([x, x], [0, 4], color="#ffd400", linewidth=1.5)
                    axis.set_xlim(extent[0], extent[1])
                    axis.set_ylim(extent[2], extent[3])
                    axis.set_xticks([])
                    axis.set_yticks([])
            axes[2 * block, 0].set_ylabel(f"Set {label}\naxial", fontsize=9)
            axes[2 * block + 1, 0].set_ylabel(f"Set {label}\ncoronal", fontsize=9)
        fig.suptitle(f"Item {number}", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(destination / f"item_{number:02d}.png", dpi=150, facecolor=surface)
        plt.close(fig)
        mapping = {"H": "A" if high_first else "B", "L": "B" if high_first else "A"}
        opposite = {"H": "L", "L": "H"}
        answers = {}
        for rating, side in item["predicted"].items():
            if rating in QUESTIONS:
                question, same_direction = QUESTIONS[rating]
                answers[question] = mapping[side if same_direction else opposite[side]]
        key.append({"item": number, "kind": item["kind"], "site": item["site"], "feature": item["feature"], "high_set": mapping["H"], "predicted_answers": answers,
                    "set_A": sets["A"].identifier.tolist(), "set_B": sets["B"].identifier.tolist()})
    write_json(destination / "key.json", {"items": key, "created_at": timestamp(), "config_sha256": sha256(CONFIG_PATH), "code_sha256": sha256(Path(__file__))})
    questions = settings["questions"]
    pd.DataFrame([{"item": number, "question": question, "answer": ""} for number in range(1, len(items) + 1) for question in questions]).to_csv(destination / "answer_sheet.csv", index=False)
    print("TASK", len(items), "items", flush=True)


def score(answers_path):
    _, _, run, _, _ = directories()
    key = read_json(run / "task/key.json")["items"]
    answers = pd.read_csv(answers_path)
    assert set(answers.columns) >= {"item", "question", "answer"}
    answers["answer"] = answers.answer.fillna("").astype(str).str.strip()
    lookup = {(int(row.item), row.question): row.answer for row in answers.itertuples()}
    rows = []
    for item in key:
        reported = {question: lookup[(item["item"], question)] for question in configuration()["task"]["questions"][1:]}
        matched = {question: reported[question] == predicted for question, predicted in item["predicted_answers"].items()}
        rows.append({"item": item["item"], "kind": item["kind"], "site": item["site"], "feature": item["feature"], "predicted_questions": len(matched), "matched": sum(matched.values()),
                     "confirmed": bool(matched) and all(matched.values()), "size_answer": reported["which set has larger nodules"],
                     "differences_reported": sum(value in ["A", "B"] for value in reported.values()), "unanswered": sum(value == "" for value in reported.values())})
    table = pd.DataFrame(rows)
    destination = run / "task" / f"score_{Path(answers_path).stem}.csv"
    table.to_csv(destination, index=False)
    print(table.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["encode", "peaks", "activations", "analyze", "pages", "task", "score"], required=True)
    parser.add_argument("--answers", type=Path, help="score stage: a filled copy of task/answer_sheet.csv")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--verify", type=int, help="encode stage: recompute this many LUNA25 records and compare with the cached campaign arrays")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.stage == "encode":
        encode(args.device, args.verify)
    elif args.stage == "peaks":
        peaks(args.device)
    elif args.stage == "activations":
        activations(args.device)
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "pages":
        pages(args.device)
    elif args.stage == "task":
        task(args.device)
    else:
        score(args.answers)


if __name__ == "__main__":
    main()

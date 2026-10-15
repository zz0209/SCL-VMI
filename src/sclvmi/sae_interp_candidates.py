import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.sparse.csgraph import connected_components
from scipy.stats import false_discovery_control, rankdata
from scipy.stats import t as student
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_interp_profiles import load_case, profile_members


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
SITES = ["fmcib:layer1_context", "fmcib:layer2", "vista:stage2", "vista:stage3"]
BINS = ["inside", "margin_0_4mm", "near_4_10mm", "far_over_10mm"]
WHOLE_REGIONS = ["all", "lung", "liver", "other_tissue", "high_hu", "edge_air"]


def entries(root, model, site, seed):
    rows = [row for row in read_json(root / "viewer/index.json")["dictionaries"] if row["model"] == model and row["site"] == site and row["seed"] == seed]
    found = {row["encoder"]: row for row in rows}
    assert len(rows) == 2 and sorted(found) == ["random", "trained"], (model, site, seed)
    return found


@torch.inference_mode()
def case_summaries(root, previous, model, site, rows, members, near_mm):
    dictionaries = {encoder: SpatialDictionary(root / "runs" / row["run_id"] / "dictionary.pt") for encoder, row in rows.items()}
    result = {encoder: np.zeros((len(members), dictionary.weights["W_enc"].shape[1]), dtype=np.float32) for encoder, dictionary in dictionaries.items()}
    started = time.monotonic()
    for index, row in enumerate(members.itertuples()):
        trained, random, distance, _, valid = load_case(root, previous, model, site, row.identifier)
        flat_valid = valid.reshape(-1)
        near = torch.from_numpy(distance.reshape(-1)[flat_valid] <= near_mm).cuda()
        for encoder, values in [("trained", trained), ("random", random)]:
            tokens = torch.from_numpy(values.reshape(values.shape[0], -1).T[flat_valid]).cuda()
            if near.any():
                result[encoder][index] = dictionaries[encoder].encode(tokens)[near].amax(0).cpu().numpy()
        if (index + 1) % 200 == 0 or index + 1 == len(members):
            elapsed = time.monotonic() - started
            print(f"{model}:{site} summaries {index + 1}/{len(members)} patients, {elapsed:.0f} s, {(index + 1) / elapsed:.1f} patients/s", flush=True)
    return result


def ranks(values):
    return rankdata(values, axis=0)


def residual_ranks(values, covariates):
    design = np.column_stack([np.ones(len(values)), ranks(covariates)])
    ranked = ranks(values)
    return ranked - design @ (np.linalg.pinv(design) @ ranked)


def correlate(left, right):
    # Rank residuals of a constant column are rounding noise far below 1e-6.
    left_norm, right_norm = np.linalg.norm(left, axis=0), np.linalg.norm(right, axis=0)
    valid = (left_norm > 1e-6)[:, None] & (right_norm > 1e-6)[None, :]
    return np.divide(left.T @ right, np.outer(left_norm, right_norm), out=np.full(valid.shape, np.nan), where=valid)


def partial(values, labels, covariates):
    return correlate(residual_ranks(values, covariates), residual_ranks(labels[:, None].astype(np.float64), covariates))[:, 0]


def p_values(correlation, count, covariates):
    freedom = count - 2 - covariates
    statistic = correlation * np.sqrt(freedom / np.clip(1 - correlation ** 2, 1e-12, None))
    return 2 * student.sf(np.abs(statistic), freedom)


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def whole_statistics(directory):
    sums, counts = [], []
    for case in [0, 2, 4, 6]:
        path = directory / f"case_{case:02d}.h5"
        assert path.with_suffix(".json").exists(), path
        with h5py.File(path, "r") as saved:
            assert list(saved.attrs["regions"]) == WHOLE_REGIONS
            sums.append(saved["region_sums"][...].astype(np.float64))
            counts.append(saved["region_counts"][...].astype(np.float64))
    sums, counts = np.stack(sums), np.stack(counts)
    means = np.divide(sums, counts[:, None, :], out=np.zeros_like(sums), where=counts[:, None, :] > 0).mean(0)
    active = sums[:, :, 0] > 0
    cases = active.sum(0)

    def share(region):
        values = np.divide(sums[:, :, region], sums[:, :, 0], out=np.zeros_like(sums[:, :, 0]), where=active)
        return np.divide(values.sum(0), cases, out=np.full(cases.shape, np.nan), where=cases > 0)

    return {"whole_lung_enrichment": ratio(means[:, 1], means[:, 3]), "whole_liver_enrichment": ratio(means[:, 2], means[:, 3]),
            "whole_high_hu_share": share(4), "whole_edge_air_share": share(5)}


def percentile(values, reference):
    clean = np.sort(reference[np.isfinite(reference)])
    return np.where(np.isfinite(values), np.searchsorted(clean, values, side="right") / len(clean) * 100, np.nan)


def seed_agreement(stability, profiles, encoder, position, reference_train, threshold, cosine):
    seeds = [int(seed) for seed in stability["seeds"]]
    agreement = np.zeros(len(reference_train), dtype=int)
    matched_seeds = np.zeros(len(reference_train), dtype=int)
    for target, seed in enumerate(seeds):
        if target == position:
            continue
        good = stability["best_cosine"][position, target] >= cosine
        matched = stability["matched"][position, target]
        train = profiles[f"{encoder}_{seed}_partial_train"][matched]
        development = profiles[f"{encoder}_{seed}_partial_development"][matched]
        sign = np.sign(reference_train)
        agreement += good & (np.sign(train) == sign) & (np.abs(train) >= threshold) & (np.sign(development) == sign)
        matched_seeds += good
    return agreement, matched_seeds


def incremental_auroc(summary, size, labels, train, development, bootstrap, generator):
    def scores(columns):
        design = np.column_stack(columns)
        mean, scale = design[train].mean(0), design[train].std(0)
        design = (design - mean) / np.where(scale > 0, scale, 1)
        model = LogisticRegression(max_iter=2000).fit(design[train], labels[train])
        return model.decision_function(design[development])

    base = scores([size])
    combined = scores([size, np.log1p(summary)])
    target = labels[development]
    differences = []
    for _ in range(bootstrap):
        sample = generator.integers(0, len(target), len(target))
        if target[sample].min() == target[sample].max():
            continue
        differences.append(roc_auc_score(target[sample], combined[sample]) - roc_auc_score(target[sample], base[sample]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return roc_auc_score(target, base), roc_auc_score(target, combined), low, high


def families(values, selected, threshold):
    result = np.full(values.shape[1], -1)
    index = np.flatnonzero(selected)
    if len(index) == 0:
        return result
    correlation = np.corrcoef(ranks(values[:, index]).T) if len(index) > 1 else np.ones((1, 1))
    _, labels = connected_components(np.atleast_2d(correlation) >= threshold, directed=False)
    result[index] = labels
    return result


def analyse(encoder, values, row, config, settings, profiles, stability, threshold, random_reference, context_data):
    labels, size, train, development, references, generator = (context_data[key] for key in ["labels", "size", "train", "development", "references", "generator"])
    seeds = [int(seed) for seed in stability["seeds"]]
    position = seeds.index(row["seed"])
    prefix = f"{encoder}_{row['seed']}_"
    live = stability["development_live"][position]
    reappearance = stability["reappearance"][position].astype(int)
    stable = live & (reappearance >= config["stability"]["stable_minimum_reappearance"])
    morphology = np.column_stack([size, references[settings["morphology"]].to_numpy()])
    frame = pd.DataFrame({"feature": np.arange(values.shape[1]), "stable": stable, "reappearance": reappearance, "live": live})
    frame["frequency"] = profiles[prefix + "frequency"]
    frame["nodule_enrichment"] = profiles[prefix + "nodule_enrichment"]
    frame["enrichment_percentile"] = percentile(profiles[prefix + "nodule_enrichment"], random_reference["nodule_enrichment"])
    for index, name in enumerate(BINS):
        frame[f"region_{name}"] = profiles[prefix + "region_mean"][index]
    frame["image_r2"] = profiles[prefix + "image_r2"]
    frame["development_auroc"] = profiles[prefix + "development_auroc"]
    frame["partial_discovery"] = partial(values[train], labels[train], size[train, None])
    frame["partial_confirmation"] = partial(values[development], labels[development], size[development, None])
    frame["p_confirmation"] = p_values(frame.partial_confirmation.to_numpy(), development.sum(), 1)
    frame["partial_morphology_discovery"] = partial(values[train], labels[train], morphology[train])
    frame["partial_morphology_confirmation"] = partial(values[development], labels[development], morphology[development])
    frame["p_morphology_confirmation"] = p_values(frame.partial_morphology_confirmation.to_numpy(), development.sum(), morphology.shape[1])
    agreement, matched_seeds = seed_agreement(stability, profiles, encoder, position, frame.partial_discovery.to_numpy(), threshold, config["stability"]["decoder_cosine"])
    frame["seed_agreement"], frame["matched_seeds"] = agreement, matched_seeds
    centered = ranks(values[train]) - ranks(values[train]).mean(0)
    frame["size_rho"] = correlate(centered, (ranks(size[train]) - ranks(size[train]).mean())[:, None])[:, 0]
    correlations = correlate(residual_ranks(values[train], size[train, None]), residual_ranks(references.loc[train, settings["references"]].to_numpy(), size[train, None]))
    strongest = np.argmax(np.where(np.isfinite(correlations), np.abs(correlations), -1), axis=1)
    frame["top_reference"] = [settings["references"][index] for index in strongest]
    frame["top_reference_partial_rho"] = correlations[np.arange(len(strongest)), strongest]
    frame.loc[~np.isfinite(frame.top_reference_partial_rho), "top_reference"] = ""
    for key, value in context_data["whole"][encoder].items():
        frame[key] = value
    selected = stable & (np.abs(frame.partial_discovery.to_numpy()) >= threshold)
    frame["discovery_selected"] = selected
    frame["q_confirmation"] = np.nan
    if selected.any():
        # A feature with no confirmation variance has no evidence and enters the correction with p = 1.
        frame.loc[selected, "q_confirmation"] = false_discovery_control(np.nan_to_num(frame.loc[selected, "p_confirmation"].to_numpy(), nan=1.0))
    same_sign = np.sign(frame.partial_confirmation) == np.sign(frame.partial_discovery)
    frame["confirmed"] = selected & same_sign.to_numpy() & (frame.q_confirmation < settings["false_discovery_rate"]).to_numpy()
    frame["beyond_morphology"] = frame.confirmed & (np.sign(frame.partial_morphology_confirmation) == np.sign(frame.partial_discovery)) & (frame.p_morphology_confirmation < 0.05)
    frame["family"] = families(values[train], frame.confirmed.to_numpy(), settings["family_correlation"])
    for column in ["size_only_auroc", "size_feature_auroc", "delta_auroc_low", "delta_auroc_high"]:
        frame[column] = np.nan
    for feature in np.flatnonzero(selected):
        frame.loc[feature, ["size_only_auroc", "size_feature_auroc", "delta_auroc_low", "delta_auroc_high"]] = incremental_auroc(values[:, feature], size, labels, train, development, settings["bootstrap"], generator)
    frame["delta_auroc"] = frame.size_feature_auroc - frame.size_only_auroc
    return frame


def site_summary(frame):
    confirmed = frame[frame.confirmed]
    return {"features": len(frame), "stable": int(frame.stable.sum()), "discovery_selected": int(frame.discovery_selected.sum()), "confirmed": len(confirmed),
            "confirmed_positive": int((confirmed.partial_discovery > 0).sum()), "confirmed_negative": int((confirmed.partial_discovery < 0).sum()),
            "beyond_morphology": int(frame.beyond_morphology.sum()), "families": int(confirmed.family.nunique()),
            "stable_enrichment_above_random_95": int((frame.stable & (frame.enrichment_percentile > 95)).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", default=",".join(SITES))
    parser.add_argument("--smoke", type=int, default=0, help="patients per split, written to a separate run directory")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = read_json(CONFIG_PATH)
    settings = config["candidates"]
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    previous = Path(storage["activations"]) / config["previous_suite"]
    output = Path(storage["runs"]) / (settings["run_id"] + (f"_smoke{args.smoke}" if args.smoke else ""))
    members = profile_members(root)
    if args.smoke:
        members = members.groupby("split", sort=False).head(args.smoke).reset_index(drop=True)
    references = pd.read_csv(root / "nodule_references.csv", dtype={"PatientID": str}).set_index("identifier").loc[members.identifier].reset_index()
    train = (members.split == "train").to_numpy()
    shared = {"labels": members.label.to_numpy(), "size": np.log(references.volume_mm3.to_numpy()), "train": train, "development": ~train, "references": references}
    code = sha256(Path(__file__))
    write_json(output / "request.json", {"run_id": output.name, "campaign": config["campaign"], "settings": settings, "stability": config["stability"], "reference_seed": config["grid"]["reference_seed"],
                                         "sites": args.sites.split(","), "members": len(members), "discovery_patients": int(train.sum()), "confirmation_patients": int((~train).sum()), "smoke": args.smoke, "code_sha256": code, "test_used": False})
    totals = {}
    for site in args.sites.split(","):
        started = time.monotonic()
        model, name = site.split(":")
        destination = output / f"{model}_{name}"
        destination.mkdir(parents=True, exist_ok=True)
        rows = entries(root, model, name, config["grid"]["reference_seed"])
        identity = {"near_mm": settings["near_mm"], "members": members.identifier.tolist(), **{encoder: {"run_id": row["run_id"], "dictionary_sha256": row["dictionary_sha256"]} for encoder, row in rows.items()}}
        if (destination / "summaries.json").exists():
            assert read_json(destination / "summaries.json") == identity, destination
            with np.load(destination / "summaries.npz") as saved:
                values = {encoder: saved[encoder] for encoder in rows}
        else:
            values = case_summaries(root, previous, model, name, rows, members, settings["near_mm"])
            np.savez_compressed(destination / "summaries.npz", **values)
            write_json(destination / "summaries.json", identity)
        with np.load(root / "profiles" / f"{model}_{name}" / "profiles.npz") as saved:
            profiles = {key: saved[key] for key in saved.files}
        checks = {}
        if not args.smoke:
            saved_members = pd.read_csv(root / "profiles" / f"{model}_{name}" / "members.csv", dtype={"PatientID": str})
            assert saved_members.identifier.tolist() == members.identifier.tolist()
            for encoder, row in rows.items():
                recomputed = partial(values[encoder][train], shared["labels"][train], shared["size"][train, None])
                stored = profiles[f"{encoder}_{row['seed']}_partial_train"]
                constant = values[encoder][train].std(0) == 0
                assert (np.isnan(recomputed) == constant).all() and np.isfinite(stored[~constant]).all(), (site, encoder)
                difference = np.abs(recomputed[~constant] - stored[~constant]).max()
                assert difference < 1e-4, (site, encoder, difference)
                checks[encoder] = {"maximum_difference": float(difference), "constant_in_discovery": int(constant.sum()), "profile_finite_for_constant": int(np.isfinite(stored[constant]).sum())}
        random_row = rows["random"]
        random_partial = partial(values["random"][train], shared["labels"][train], shared["size"][train, None])
        threshold = float(np.percentile(np.abs(random_partial[np.isfinite(random_partial)]), settings["random_percentile"]))
        random_reference = {"nodule_enrichment": profiles[f"random_{random_row['seed']}_nodule_enrichment"]}
        whole_root = Path(storage["runs"]) / config["whole_ct"]["run_id"] / "browser_cache"
        context_data = {**shared, "whole": {encoder: whole_statistics(whole_root / row["run_id"]) for encoder, row in rows.items()}}
        frames = {}
        for encoder, row in rows.items():
            with np.load(root / "stability" / row["stability_group"] / "features.npz") as saved:
                stability = {key: saved[key] for key in saved.files}
            context_data["generator"] = np.random.default_rng(settings["bootstrap_seed"])
            frames[encoder] = analyse(encoder, values[encoder], row, config, settings, profiles, stability, threshold, random_reference, context_data)
            frames[encoder].to_csv(destination / f"{encoder}_features.csv", index=False)
        candidates = frames["trained"][frames["trained"].confirmed].assign(strength=lambda frame: frame.partial_discovery.abs()).sort_values(["family", "strength"], ascending=[True, False]).drop(columns="strength")
        candidates.to_csv(destination / "candidates.csv", index=False)
        totals[site] = {"threshold": threshold, "profile_consistency": checks, "trained": site_summary(frames["trained"]), "random": site_summary(frames["random"]),
                        "runs": {encoder: row["run_id"] for encoder, row in rows.items()}, "seconds": time.monotonic() - started}
        write_json(destination / "summary.json", {**totals[site], "created_at": timestamp(), "code_sha256": code, "test_used": False})
        print(site, totals[site]["trained"], totals[site]["random"], flush=True)
    write_json(output / "summary.json", {"sites": totals, "created_at": timestamp(), "code_sha256": code, "test_used": False})


if __name__ == "__main__":
    main()

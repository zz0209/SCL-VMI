import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_context import VARIABLES, stable_value


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
BINS = [("inside", -np.inf, 0.0), ("margin_0_4mm", 0.0, 4.0), ("near_4_10mm", 4.0, 10.0), ("far_over_10mm", 10.0, np.inf)]
SITE_KEYS = {
    ("fmcib", "layer1_context"): {"trained": ("new", "context_trained"), "random": ("new", "context_random"), "distance": "context_distance", "variables": "context_variables", "valid": "context_valid", "viewer": "fmcib_context"},
    ("fmcib", "layer2"): {"trained": ("previous", "layer2"), "random": ("new", "layer2_random"), "distance": "layer2_distance", "variables": "layer2_variables", "valid": None, "viewer": "fmcib"},
    ("vista", "stage2"): {"trained": ("previous", "stage2"), "random": ("new", "stage2_random"), "distance": "stage2_distance", "variables": "stage2_variables", "valid": None, "viewer": "vista"},
    ("vista", "stage3"): {"trained": ("previous", "stage3"), "random": ("new", "stage3_random"), "distance": "stage3_distance", "variables": "stage3_variables", "valid": None, "viewer": "vista"},
}


def profile_members(root):
    members = pd.read_csv(root / "members.csv", dtype={"PatientID": str})
    viewers = pd.read_csv(root / "viewer_members.csv", dtype={"PatientID": str})
    chosen = members.loc[members.identifier.isin(viewers.identifier)]
    rest = members.loc[~members.PatientID.isin(chosen.PatientID)]
    rest = rest.assign(order=rest.identifier.map(lambda value: stable_value(f"viewer-record:{value}"))).sort_values("order").groupby("PatientID", sort=False).head(1).drop(columns="order")
    frame = pd.concat([chosen, rest]).sort_values(["split", "identifier"]).reset_index(drop=True)
    assert frame.PatientID.is_unique and set(frame.PatientID) == set(members.PatientID)
    return frame


def load_case(root, previous, model, site, identifier):
    keys = SITE_KEYS[(model, site)]
    with np.load(root / "cases" / model / f"{identifier}.npz") as new, np.load(previous / model / "cases" / f"{identifier}.npz") as old:
        source = {"new": new, "previous": old}
        trained = source[keys["trained"][0]][keys["trained"][1]]
        random = source[keys["random"][0]][keys["random"][1]]
        distance = new[keys["distance"]]
        variables = new[keys["variables"]]
        valid = new[keys["valid"]] if keys["valid"] else np.ones(distance.shape, dtype=bool)
    return trained, random, distance, variables, valid


def auroc_columns(scores, labels):
    ranks = rankdata(scores, axis=0)
    positive = labels == 1
    count = positive.sum()
    values = (ranks[positive].sum(0) - count * (count + 1) / 2) / (count * (len(labels) - count))
    return np.where(scores.std(0) > 0, values, np.nan)


def rank_matrix(values):
    return torch.from_numpy(pd.DataFrame(values).rank(axis=0).to_numpy(dtype=np.float64))


def partial_rank_correlation(summary, labels, size):
    ranked = rank_matrix(summary)
    covariates = torch.stack([torch.ones(len(labels), dtype=torch.float64), rank_matrix(size[:, None])[:, 0]], 1)
    projection = torch.eye(len(labels), dtype=torch.float64) - covariates @ torch.linalg.pinv(covariates)
    residual_summary = projection @ ranked
    residual_label = projection @ rank_matrix(labels[:, None].astype(float))[:, 0]
    numerator = (residual_summary * residual_label[:, None]).sum(0)
    denominator = residual_summary.norm(dim=0) * residual_label.norm()
    return torch.where(denominator > 0, numerator / denominator, torch.full_like(numerator, float("nan"))).numpy()


@torch.inference_mode()
def profiles(root, model, site, group, seeds, variant=None, samples_per_case=16, smoke=0):
    config = read_json(CONFIG_PATH)
    storage, _, _ = context()
    previous = Path(storage["activations"]) / config["previous_suite"]
    members = profile_members(root)
    if smoke:
        members = members.groupby("split", sort=False).head(smoke).reset_index(drop=True)
    references = pd.read_csv(root / "nodule_references.csv", dtype={"PatientID": str}).set_index("identifier")
    viewers = pd.read_csv(root / "viewer_members.csv", dtype={"PatientID": str})
    viewer_index = {value: index for index, value in enumerate(viewers.identifier)}
    runs = {}
    for encoder in ["trained", "random"]:
        for seed in seeds:
            run = f"{config['campaign']}_{encoder}_{model}_{site}_{group}_s{seed}" + (f"_{variant}" if variant else "")
            runs[(encoder, seed)] = SpatialDictionary(root / "runs" / run / "dictionary.pt")
    reference = config["grid"]["reference_seed"]
    features = runs[("trained", reference)].weights["W_enc"].shape[1]
    count = len(members)
    region_sums = {key: torch.zeros((len(BINS), features), dtype=torch.float64, device="cuda") for key in runs}
    region_counts = torch.zeros((count, len(BINS)), dtype=torch.float64)
    active_fraction = {key: torch.zeros(features, dtype=torch.float64, device="cuda") for key in runs}
    summary = {key: torch.zeros((count, features), dtype=torch.float32, device="cuda") for key in runs}
    viewer_maximum = {key: torch.zeros((len(viewers), features), dtype=torch.float32, device="cuda") for key in runs}
    sampled_variables, sampled_activity = [], {key: [] for key in [("trained", reference), ("random", reference)]}
    sample_split = []
    started = time.monotonic()
    for index, row in enumerate(members.itertuples()):
        trained, random, distance, variables, valid = load_case(root, previous, model, site, row.identifier)
        flat_valid = valid.reshape(-1)
        flat_distance = distance.reshape(-1)[flat_valid]
        assert len(flat_distance) > 0, row.identifier
        tokens = {"trained": torch.from_numpy(trained.reshape(trained.shape[0], -1).T[flat_valid]).cuda(), "random": torch.from_numpy(random.reshape(random.shape[0], -1).T[flat_valid]).cuda()}
        bins = [(flat_distance >= low) & (flat_distance < high) for _, low, high in BINS]
        region_counts[index] = torch.tensor([value.sum() for value in bins], dtype=torch.float64)
        near = torch.from_numpy(flat_distance <= 4.0).cuda()
        generator = np.random.default_rng(stable_value(f"profile-sample:{row.identifier}"))
        chosen = generator.choice(len(flat_distance), size=min(samples_per_case, len(flat_distance)), replace=False)
        sampled_variables.append(np.concatenate([variables.reshape(len(VARIABLES), -1).T[flat_valid][chosen], np.clip(flat_distance[chosen], -20, 20)[:, None]], 1))
        sample_split.append(np.full(len(chosen), row.split == "train"))
        for key, dictionary in runs.items():
            values = dictionary.encode(tokens[key[0]])
            for bin_index, mask in enumerate(bins):
                if mask.any():
                    region_sums[key][bin_index] += values[torch.from_numpy(mask).cuda()].double().sum(0)
            active_fraction[key] += (values > 0).double().mean(0)
            summary[key][index] = values[near].amax(0) if near.any() else torch.zeros(features, device="cuda")
            if row.identifier in viewer_index:
                viewer_maximum[key][viewer_index[row.identifier]] = values.amax(0)
            if key in sampled_activity:
                sampled_activity[key].append(values[torch.from_numpy(chosen).cuda()].cpu())
        if index % 100 == 0:
            write_json(root / "progress" / f"profiles-{model}-{site}.json", {"task": "profiles", "completed": index + 1, "total": count, "elapsed_seconds": time.monotonic() - started, "updated_at": timestamp()})
    labels = members.label.to_numpy()
    size = np.log(references.loc[members.identifier, "volume_mm3"].to_numpy())
    train = (members.split == "train").to_numpy()
    development = ~train
    size_auroc = float(auroc_columns(size[development, None], labels[development])[0])
    variables_matrix = np.concatenate(sampled_variables)
    sample_train = np.concatenate(sample_split)
    mean, scale = variables_matrix[sample_train].mean(0), variables_matrix[sample_train].std(0)
    design = np.concatenate([np.ones((len(variables_matrix), 1)), (variables_matrix - mean) / np.where(scale > 0, scale, 1)], 1)
    design = torch.from_numpy(design).cuda()
    output = {}
    for key in runs:
        weights = region_counts.cuda()
        mean_by_bin = (region_sums[key] / weights.sum(0)[:, None]).cpu().numpy()
        overall = (region_sums[key].sum(0) / weights.sum()).cpu().numpy()
        frequency = (active_fraction[key] / count).cpu().numpy()
        values = summary[key].cpu().numpy().astype(np.float64)
        partial_train = partial_rank_correlation(values[train], labels[train], size[train])
        partial_development = partial_rank_correlation(values[development], labels[development], size[development])
        auroc = auroc_columns(values[development], labels[development])
        entry = {
            "region_mean": mean_by_bin.astype(np.float32), "overall_mean": overall.astype(np.float32),
            "nodule_enrichment": ((mean_by_bin[0] * region_counts[:, 0].sum().item() + mean_by_bin[1] * region_counts[:, 1].sum().item()) / max(region_counts[:, :2].sum().item(), 1) / np.where(overall > 0, overall, np.nan)).astype(np.float32),
            "frequency": frequency.astype(np.float32), "partial_train": partial_train.astype(np.float32), "partial_development": partial_development.astype(np.float32),
            "development_auroc": auroc.astype(np.float32), "viewer_maximum": viewer_maximum[key].cpu().numpy(),
        }
        if key in sampled_activity:
            activity = torch.cat(sampled_activity[key]).double().cuda()
            fitted_train = design[torch.from_numpy(sample_train).cuda()]
            penalty = torch.eye(design.shape[1], dtype=torch.float64, device="cuda")
            penalty[0, 0] = 0
            coefficients = torch.linalg.solve(fitted_train.T @ fitted_train + 1.0 * penalty, fitted_train.T @ activity[torch.from_numpy(sample_train).cuda()])
            target = activity[torch.from_numpy(~sample_train).cuda()]
            prediction = design[torch.from_numpy(~sample_train).cuda()] @ coefficients
            residual = (target - prediction).square().sum(0)
            total = (target - target.mean(0)).square().sum(0)
            entry["image_r2"] = torch.where(total > 0, 1 - residual / total, torch.full_like(total, float("nan"))).cpu().numpy().astype(np.float32)
        output[key] = entry
    return members, output, size_auroc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--smoke", type=int, default=0, help="patients per split; uses the reference seed only")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = read_json(CONFIG_PATH)
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    seeds = [config["grid"]["reference_seed"]] if args.smoke else config["grid"]["seeds"]
    destination = root / ("profiles_smoke" if args.smoke else "profiles") / f"{args.model}_{args.site}"
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    members, output, size_auroc = profiles(root, args.model, args.site, args.group, seeds, args.variant, smoke=args.smoke)
    arrays = {}
    for (encoder, seed), entry in output.items():
        for name, value in entry.items():
            arrays[f"{encoder}_{seed}_{name}"] = value
    np.savez_compressed(destination / "profiles.npz", **arrays)
    members.to_csv(destination / "members.csv", index=False)
    write_json(destination / "summary.json", {
        "model": args.model, "site": args.site, "group": args.group, "variant": args.variant, "seeds": seeds, "bins": [name for name, _, _ in BINS],
        "variables": VARIABLES + ["signed_distance_clipped_20mm"], "members": len(members), "discovery_patients": int((members.split == "train").sum()), "confirmation_patients": int((members.split == "development").sum()),
        "size_only_development_auroc": size_auroc, "summary_rule": "maximum activation over valid positions within 4 mm of the target nodule surface or inside it",
        "partial_correlation": "Spearman partial correlation between the case summary and malignancy, controlling log nodule volume",
        "image_r2": "ridge (alpha 1) fit on 16 sampled valid positions per discovery case, R2 on confirmation positions; reference seeds only",
        "profiles_sha256": sha256(destination / "profiles.npz"), "code_sha256": sha256(Path(__file__)), "seconds": time.monotonic() - started, "created_at": timestamp(), "test_used": False,
    })
    print("PROFILES", args.model, args.site, size_auroc, flush=True)


if __name__ == "__main__":
    main()

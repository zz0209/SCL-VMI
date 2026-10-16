import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_interp_profiles import load_case, profile_members


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"


def region_tokens(root, previous, model, site, encoder, identifiers, regions):
    collected = {name: [] for name, _, _ in regions}
    patients = {name: [] for name, _, _ in regions}
    for index, identifier in enumerate(identifiers):
        trained, random, distance, _, valid = load_case(root, previous, model, site, identifier)
        values = trained if encoder == "trained" else random
        flat_valid = valid.reshape(-1)
        tokens = values.reshape(values.shape[0], -1).T[flat_valid]
        flat_distance = distance.reshape(-1)[flat_valid]
        for name, low, high in regions:
            mask = (flat_distance >= low) & (flat_distance < high)
            collected[name].append(tokens[mask])
            patients[name].append(np.full(mask.sum(), index))
    return {name: (torch.from_numpy(np.concatenate(parts)).cuda(), torch.from_numpy(np.concatenate(patients[name])).cuda()) for name, parts in collected.items()}


@torch.inference_mode()
def coverage(dictionary, tokens, stable, nodule_region, thresholds, patient_count, bootstrap, seed):
    result, sums, counts, per_patient = {}, {}, {}, {}
    keep = torch.from_numpy(stable).cuda().float()
    for name, (values, patients) in tokens.items():
        normalized = (values - dictionary.mean) / dictionary.scale
        codes = dictionary.encode(values)
        deviation = (normalized - normalized.mean(0)).double().square().sum(1)
        full = (codes @ dictionary.weights["W_dec"] + dictionary.weights["b_dec"] - normalized).double().square().sum(1)
        kept = ((codes * keep) @ dictionary.weights["W_dec"] + dictionary.weights["b_dec"] - normalized).double().square().sum(1)
        result[name] = {"tokens": len(values), "explained_variance": float(1 - full.sum() / deviation.sum()), "stable_explained_variance": float(1 - kept.sum() / deviation.sum()),
                        "stable_activation_share": float((codes * keep).sum() / codes.sum())}
        sums[name], counts[name] = codes.double().sum(0), len(values)
        per_patient[name] = [torch.zeros(patient_count, dtype=torch.float64, device="cuda").index_add_(0, patients, value) for value in [kept, deviation]]
    # Patient bootstrap of the nodule-minus-far difference in stable-feature explained variance (region means held fixed).
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.zeros((bootstrap, patient_count), dtype=torch.float64, device="cuda")
    weights.scatter_add_(1, torch.randint(patient_count, (bootstrap, patient_count), generator=generator, device="cuda"), torch.ones((bootstrap, patient_count), dtype=torch.float64, device="cuda"))
    sampled = {name: 1 - (weights @ kept) / (weights @ deviation) for name, (kept, deviation) in per_patient.items()}
    difference = (sampled[nodule_region] - sampled["far_over_10mm"]).cpu().numpy()
    interval = {"nodule_minus_far_low": float(np.percentile(difference, 2.5)), "nodule_minus_far_high": float(np.percentile(difference, 97.5)), "bootstrap": bootstrap}
    # Nodule enrichment: mean activation at the nodule region over the mean over all evaluated positions.
    overall = sum(sums.values()) / sum(counts.values())
    enrichment = (sums[nodule_region] / counts[nodule_region] / torch.where(overall > 0, overall, torch.ones_like(overall))).cpu().numpy()
    enrichment[overall.cpu().numpy() == 0] = 0
    enriched = {str(value): {"features": int((enrichment >= value).sum()), "stable": int(((enrichment >= value) & stable).sum())} for value in thresholds}
    return result, enriched, interval


def evaluate(root, previous, group, stability_directory, regions, identifiers):
    summary = read_json(stability_directory / group / "summary.json")
    with np.load(stability_directory / group / "features.npz") as saved:
        seeds = [int(seed) for seed in saved["seeds"]]
        live = saved["development_live"]
        reappearance = saved["reappearance"]
    minimum = len(seeds) - 1 if stability_directory.name == "stability_budget" else read_json(CONFIG_PATH)["stability"]["stable_minimum_reappearance"]
    first = next(iter(summary["dictionaries"]))
    dictionary = SpatialDictionary(root / "runs" / first / "dictionary.pt")
    request = dictionary.request
    encoder = "random" if "_random_" in first else "trained"
    assert request["seed"] == seeds[0]
    stable = live[0] & (reappearance[0] >= minimum)
    tokens = region_tokens(root, previous, request["model"], request["site"], encoder, identifiers, regions)
    settings = read_json(CONFIG_PATH)["nodule_coverage"]
    regions_result, enriched, interval = coverage(dictionary, tokens, stable, regions[0][0], settings["enrichment_thresholds"], len(identifiers), settings["bootstrap"], settings["bootstrap_seed"])
    return {"group": group, "run_id": first, "seeds": seeds, "stable_rule": f"reappears in at least {minimum} of {len(seeds) - 1} other seeds", "stable_features": int(stable.sum()),
            "regions": regions_result, "nodule_enriched": enriched, "interval": interval}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True, help="comma-separated stability group names")
    parser.add_argument("--directory", default="stability_budget", choices=["stability_budget", "stability"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    config = read_json(CONFIG_PATH)
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    previous = Path(storage["activations"]) / config["previous_suite"]
    regions = [tuple(item) for item in config["nodule_coverage"]["regions"]]
    members = profile_members(root)
    identifiers = members.loc[members.split == "development", "identifier"].tolist()
    destination = root / "coverage"
    destination.mkdir(exist_ok=True)
    for group in args.groups.split(","):
        started = time.monotonic()
        result = evaluate(root, previous, group, root / args.directory, regions, identifiers)
        regions_result = result["regions"]
        result.update({"patients": len(identifiers), "patient_rule": "one record per development patient (profile members)", "region_rule": "signed distance to the target nodule surface in mm, lower bound inclusive",
                       "nodule_minus_far": regions_result["nodule_0_4mm"]["stable_explained_variance"] - regions_result["far_over_10mm"]["stable_explained_variance"],
                       "code_sha256": sha256(Path(__file__)), "seconds": time.monotonic() - started, "created_at": timestamp(), "test_used": False})
        write_json(destination / f"{args.directory}_{group}.json", result)
        print("COVERAGE", group, result["stable_features"], {name: round(value["stable_explained_variance"], 3) for name, value in regions_result.items()}, round(result["nodule_minus_far"], 3), [round(result["interval"]["nodule_minus_far_low"], 3), round(result["interval"]["nodule_minus_far_high"], 3)], result["nodule_enriched"], flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

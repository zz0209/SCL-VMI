import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .context import ROOT, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"


def group_key(job):
    key = f"{job['encoder']}_{job['model']}_{job['site']}_e{job['expansion']}_k{job['k']}"
    return f"{key}_{job['variant']}" if "variant" in job else key


@torch.inference_mode()
def pca_fvu(matrix, sample, dictionary, rank, per_patient):
    train = (development_sample(matrix, per_patient, "train") - dictionary.mean) / dictionary.scale
    mean = train.mean(0, keepdim=True)
    _, _, basis = torch.linalg.svd((train - mean).double(), full_matrices=False)
    components = basis[:rank]
    test = ((sample - dictionary.mean) / dictionary.scale).double()
    centered = test - mean.double()
    residual = centered - (centered @ components.T) @ components
    return float(residual.square().sum() / (test - test.mean(0)).square().sum())


def development_sample(matrix, per_patient, split="development"):
    values = np.load(matrix / f"{split}.npy", mmap_mode="r")
    offsets = np.load(matrix / f"{split}_offsets.npy")
    rows = np.concatenate([np.arange(start, min(end, start + per_patient)) for start, end in zip(offsets[:-1], offsets[1:])])
    return torch.from_numpy(np.asarray(values[rows], dtype=np.float32)).cuda()


@torch.inference_mode()
def standardized(values):
    centered = values - values.mean(0, keepdim=True)
    norm = centered.norm(dim=0, keepdim=True)
    return centered / torch.where(norm > 0, norm, torch.ones_like(norm)), norm[0] > 0


@torch.inference_mode()
def fvu(sample, dictionary, activations, keep=None):
    normalized = (sample - dictionary.mean) / dictionary.scale
    codes = activations if keep is None else activations * keep[None]
    recovered = codes @ dictionary.weights["W_dec"] + dictionary.weights["b_dec"]
    return float((recovered - normalized).double().square().sum() / (normalized - normalized.mean(0)).double().square().sum())


@torch.inference_mode()
def analyze(root, jobs, config):
    settings = config["stability"]
    reference = config["grid"]["reference_seed"]
    jobs = sorted(jobs, key=lambda job: (job["seed"] != reference, job["seed"]))
    seeds = [job["seed"] for job in jobs]
    dictionaries = [SpatialDictionary(root / "runs" / job["run_id"] / "dictionary.pt") for job in jobs]
    matrix = root / jobs[0]["matrix_directory"] / jobs[0]["model"] / jobs[0]["site"]
    sample = development_sample(matrix, settings["development_tokens_per_patient"])
    decoders = [torch.nn.functional.normalize(item.weights["W_dec"], dim=1) for item in dictionaries]
    activations = [item.encode(sample) for item in dictionaries]
    standardized_values = [standardized(values) for values in activations]
    features = decoders[0].shape[0]
    count = len(seeds)
    best_cosine = np.zeros((count, count, features), dtype=np.float32)
    matched = np.zeros((count, count, features), dtype=np.int32)
    matched_correlation = np.full((count, count, features), np.nan, dtype=np.float32)
    best_correlation = np.full((count, count, features), np.nan, dtype=np.float32)
    for source in range(count):
        for target in range(count):
            if source == target:
                continue
            cosine = decoders[source] @ decoders[target].T
            value, index = cosine.max(1)
            best_cosine[source, target], matched[source, target] = value.cpu().numpy(), index.cpu().numpy()
            left, left_live = standardized_values[source]
            right, right_live = standardized_values[target]
            correlation = left.T @ right
            correlation[:, ~right_live] = -1
            pair = correlation[torch.arange(features, device=index.device), index]
            pair[~left_live | ~right_live[index]] = float("nan")
            matched_correlation[source, target] = pair.cpu().numpy()
            best = correlation.max(1).values
            best[~left_live] = float("nan")
            best_correlation[source, target] = best.cpu().numpy()
    threshold = settings["decoder_cosine"]
    reappearance = (best_cosine >= threshold).sum(1)
    live = [(values > 0).any(0).cpu().numpy() for values in activations]
    statistics = [np.load(root / "runs" / job["run_id"] / "development_feature_statistics.npz") for job in jobs]
    development_live = [item["counts"] > 0 for item in statistics]
    stable = [(reappearance[index] >= settings["stable_minimum_reappearance"]) & development_live[index] for index in range(count)]
    unstable = [(reappearance[index] <= settings["unstable_maximum_reappearance"]) & development_live[index] for index in range(count)]
    seeds_summary = []
    for index in range(count):
        keep = torch.from_numpy(stable[index]).to(sample.device).float()
        seeds_summary.append({
            "seed": seeds[index], "run_id": jobs[index]["run_id"],
            "live_features": int(development_live[index].sum()), "stable_features": int(stable[index].sum()), "unstable_features": int(unstable[index].sum()),
            "stable_fraction_of_live": float(stable[index].sum() / max(development_live[index].sum(), 1)),
            "sample_fvu": fvu(sample, dictionaries[index], activations[index]),
            "stable_feature_fvu": fvu(sample, dictionaries[index], activations[index], keep),
        })
    pairs = []
    for target in range(1, min(count, 3)):
        cosine = (decoders[0] @ decoders[target].T).cpu().numpy()
        rows, columns = linear_sum_assignment(-cosine)
        pairs.append({"seeds": [seeds[0], seeds[target]], "pw_mcc_decoder_cosine": float(cosine[rows, columns].mean())})
    subspace = []
    reference_unstable = decoders[0][torch.from_numpy(unstable[0]).cuda()]
    if len(reference_unstable) >= 4:
        _, singular, basis = torch.linalg.svd(reference_unstable, full_matrices=False)
        energy = torch.cumsum(singular ** 2, 0) / (singular ** 2).sum()
        rank = int((energy < settings["subspace_energy"]).sum()) + 1
        for target in range(1, count):
            other = decoders[target][torch.from_numpy(unstable[target]).cuda()]
            if len(other) < 4:
                continue
            _, other_singular, other_basis = torch.linalg.svd(other, full_matrices=False)
            other_energy = torch.cumsum(other_singular ** 2, 0) / (other_singular ** 2).sum()
            other_rank = int((other_energy < settings["subspace_energy"]).sum()) + 1
            projection = reference_unstable @ other_basis[:other_rank].T
            subspace.append({"seed": seeds[target], "captured_energy": float(projection.square().sum() / reference_unstable.square().sum()), "target_rank": other_rank, "chance_energy": other_rank / reference_unstable.shape[1]})
        subspace_summary = {"reference_unstable": int(len(reference_unstable)), "reference_rank_90": rank, "input_dimension": int(reference_unstable.shape[1]), "pairs": subspace, "mean_captured_energy": float(np.mean([row["captured_energy"] for row in subspace])) if subspace else None}
    else:
        subspace_summary = {"reference_unstable": int(len(reference_unstable)), "pairs": []}
    return {
        "seeds": seeds_summary, "pairs": pairs, "unstable_subspace": subspace_summary,
        "pca_reference": {"rank": jobs[0]["k"], "sample_fvu": pca_fvu(matrix, sample, dictionaries[0], jobs[0]["k"], settings["development_tokens_per_patient"]), "fit": "training-patient sample with the same per-patient count, dictionary normalization"},
        "sample_tokens": int(len(sample)), "sample_rule": f"first {settings['development_tokens_per_patient']} saved development vectors per patient",
        "reference_stable_explained_variance": 1 - seeds_summary[0]["stable_feature_fvu"],
    }, {"seeds": np.asarray(seeds), "best_cosine": best_cosine, "matched": matched, "matched_correlation": matched_correlation, "best_correlation": best_correlation, "reappearance": reappearance.astype(np.int16), "development_live": np.stack(development_live)}


def save(destination, key, jobs, completed, summary, arrays, root, started):
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "features.npz", **arrays)
    write_json(destination / "summary.json", {"group": key, **{name: jobs[0][name] for name in ["encoder", "model", "site", "k", "expansion"]}, **summary, "dictionaries": {job["run_id"]: read_json(root / "runs" / job["run_id"] / "result.json")["dictionary_sha256"] for job in completed}, "features_sha256": sha256(destination / "features.npz"), "seconds": time.monotonic() - started, "source_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})


def budget(root, config, models):
    diagnostic = config["budget_diagnostic"]
    seeds = diagnostic["seeds"]
    restricted = {**config, "stability": {**config["stability"], "stable_minimum_reappearance": len(seeds) - 1, "unstable_maximum_reappearance": 0}}
    groups = defaultdict(list)
    for path in sorted((root / "jobs").glob("*.json")):
        job = read_json(path)
        if job["encoder"] == "trained" and job["seed"] in seeds:
            groups[group_key(job)].append(job)
    for variant in diagnostic["variants"]:
        for model, site, expansion, k in variant["configurations"]:
            if model not in models:
                continue
            base = f"trained_{model}_{site}_e{expansion}_k{k}"
            for key in [base, f"{base}_{variant['name']}"]:
                destination = root / "stability_budget" / key
                if (destination / "summary.json").exists():
                    continue
                completed = [job for job in groups[key] if (root / "runs" / job["run_id"] / "result.json").exists()]
                if len(completed) < len(seeds):
                    print("WAITING", key, len(completed), flush=True)
                    continue
                started = time.monotonic()
                summary, arrays = analyze(root, completed, restricted)
                save(destination, key, groups[key], completed, {**summary, "seeds_used": seeds, "stable_rule": f"reappears in all {len(seeds) - 1} other seeds"}, arrays, root, started)
                print("BUDGET", key, summary["reference_stable_explained_variance"], [item["stable_features"] for item in summary["seeds"]], flush=True)
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--models", default="fmcib,vista")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.budget:
        budget(args.root, config, args.models.split(","))
        return
    total = len(config["grid"]["seeds"])
    while True:
        groups = defaultdict(list)
        for path in sorted((args.root / "jobs").glob("*.json")):
            job = read_json(path)
            groups[group_key(job)].append(job)
        pending = 0
        for key, jobs in sorted(groups.items()):
            destination = args.root / "stability" / key
            if (destination / "summary.json").exists():
                continue
            pending += 1
            completed = [job for job in jobs if (args.root / "runs" / job["run_id"] / "result.json").exists()]
            if len(completed) < total:
                continue
            started = time.monotonic()
            summary, arrays = analyze(args.root, completed, config)
            save(destination, key, jobs, completed, summary, arrays, args.root, started)
            print("STABILITY", key, summary["reference_stable_explained_variance"], summary["seeds"][0]["stable_features"], flush=True)
            torch.cuda.empty_cache()
        if not args.wait or pending == 0:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()

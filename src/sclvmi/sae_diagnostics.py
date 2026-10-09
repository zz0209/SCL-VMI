import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .context import context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_data import configuration
from .sae_evaluation import prediction_metrics


def bootstrap_predictions(run, repetitions=1000):
    predictions = pd.read_csv(run / "downstream_full/predictions.csv", dtype={"PatientID": str})
    groups = [group.index.to_numpy() for _, group in predictions.groupby("PatientID", sort=True)]
    generator = np.random.default_rng(20261008)
    values = []
    for _ in tqdm(range(repetitions), desc=f"{run.name} patient bootstrap", mininterval=3):
        indices = np.concatenate([groups[index] for index in generator.integers(len(groups), size=len(groups))])
        sampled = predictions.iloc[indices]
        if sampled.label.nunique() < 2:
            continue
        metrics = prediction_metrics(sampled)
        values.append([metrics["probability_mae"], metrics["recovered_auroc"] - metrics["original_auroc"], metrics["recovered_ap"] - metrics["original_ap"]])
    values = np.asarray(values)
    assert len(values) >= repetitions * 0.99
    names = ["probability_mae", "auroc_change", "ap_change"]
    result = {"method": "paired patient bootstrap, percentile 95% intervals", "replicates": len(values), "seed": 20261008, "intervals": {name: np.quantile(values[:, index], [0.025, 0.975]).tolist() for index, name in enumerate(names)}, "completed_at": timestamp()}
    np.save(run / "downstream_full/bootstrap_values.npy", values)
    write_json(run / "downstream_full/bootstrap.json", result)
    return result


@torch.inference_mode()
def compare_seeds(runs, destination, samples_per_patient=20):
    dictionaries = [SpatialDictionary(run / "dictionary.pt") for run in runs]
    requests = [dictionary.request for dictionary in dictionaries]
    for request in requests[1:]:
        for key in ["model", "site", "dict_size", "k", "matrix_sha256"]:
            assert request[key] == requests[0][key]
    matrix = Path(requests[0]["matrix"])
    source = np.load(matrix / "development.npy", mmap_mode="r")
    tokens_per_patient = configuration()["sampling"]["development_tokens_per_patient"]
    assert len(source) % tokens_per_patient == 0
    assert 1 <= samples_per_patient <= tokens_per_patient
    generator = np.random.default_rng(20261008)
    indices = np.concatenate([start + generator.choice(tokens_per_patient, samples_per_patient, replace=False) for start in range(0, len(source), tokens_per_patient)])
    tokens = torch.from_numpy(np.array(source[indices], copy=True)).cuda()
    encoded = []
    decoders = []
    coverage = []
    for dictionary in tqdm(dictionaries, desc="Cross-seed activation encoding", mininterval=1):
        values = torch.cat([dictionary.encode(part) for part in tokens.split(1024)])
        counts = (values > 0).sum(0)
        coverage.append({"seed": dictionary.request["seed"], "activation_count_quantiles": torch.quantile(counts.float(), torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=counts.device)).cpu().tolist(), "zero_activation_features": int((counts == 0).sum()), "features_active_below_0_1_percent": int((counts < len(tokens) * 0.001).sum())})
        values -= values.mean(0)
        norms = values.norm(dim=0)
        nonconstant = norms > 1e-8
        values[:, nonconstant] /= norms[nonconstant]
        values[:, ~nonconstant] = 0
        encoded.append(values)
        decoders.append(torch.nn.functional.normalize(dictionary.weights["W_dec"], dim=1))
    comparisons = []
    for first, second in tqdm(list(combinations(range(len(runs)), 2)), desc="Cross-seed feature matching", mininterval=1):
        correlation = (encoded[first].T @ encoded[second]).cpu().numpy()
        source_index, target_index = linear_sum_assignment(correlation, maximize=True)
        matched = correlation[source_index, target_index]
        decoder_cosine = (decoders[first][source_index] * decoders[second][target_index]).sum(1).cpu().numpy()
        nearest = correlation.argmax(1)
        reciprocal = correlation.argmax(0)[nearest] == np.arange(len(nearest))
        pair_name = f"s{requests[first]['seed']}_s{requests[second]['seed']}"
        np.savez(destination / f"{pair_name}.npz", source_index=source_index, target_index=target_index, correlation=matched, decoder_cosine=decoder_cosine)
        comparisons.append({"pair": pair_name, "matched_correlation_quantiles": np.quantile(matched, [0, 0.1, 0.5, 0.9, 1]).tolist(), "fraction_correlation_ge_0_8": float(np.mean(matched >= 0.8)), "fraction_correlation_ge_0_5": float(np.mean(matched >= 0.5)), "matched_decoder_cosine_median": float(np.median(decoder_cosine)), "reciprocal_nearest_fraction": float(np.mean(reciprocal))})
    result = {"method": "maximum-total activation-correlation assignment; feature identifiers remain dictionary-local", "tokens": len(tokens), "tokens_per_patient": samples_per_patient, "patients": len(source) // tokens_per_patient, "runs": [run.name for run in runs], "coverage": coverage, "comparisons": comparisons, "source_sha256": sha256(Path(__file__)), "completed_at": timestamp(), "test_used": False}
    write_json(destination / "seed_comparison.json", result)
    return result


@torch.inference_mode()
def pca_reference(run, destination):
    request = read_json(run / "request.json")
    matrix = Path(request["matrix"])
    source = np.load(matrix / "train.npy", mmap_mode="r")
    normalization = np.load(matrix / "normalization.npz")
    mean = torch.tensor(normalization["mean"], device="cuda", dtype=torch.float64)
    scale = float(normalization["scale"])
    covariance = torch.zeros((source.shape[1], source.shape[1]), dtype=torch.float64, device="cuda")
    for start in tqdm(range(0, len(source), 8192), desc="PCA training covariance", mininterval=3):
        batch = (torch.tensor(np.array(source[start:start + 8192], copy=True), device="cuda", dtype=torch.float64) - mean) / scale
        covariance += batch.T @ batch
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance / len(source))
    rank = request["k"]
    basis = eigenvectors[:, -rank:].float()
    source = np.load(matrix / "development.npy", mmap_mode="r")
    mean = mean.float()
    sum_x = torch.zeros_like(mean, dtype=torch.float64)
    sum_square = 0.0
    error = 0.0
    cosine_sum = 0.0
    for start in tqdm(range(0, len(source), 8192), desc="PCA development reconstruction", mininterval=3):
        raw = torch.tensor(np.array(source[start:start + 8192], copy=True), device="cuda")
        batch = (raw - mean) / scale
        recovered = batch @ basis @ basis.T
        sum_x += batch.double().sum(0)
        sum_square += float(batch.double().square().sum())
        error += float((recovered - batch).double().square().sum())
        cosine_sum += float(torch.nn.functional.cosine_similarity(recovered * scale + mean, raw, dim=1).double().sum())
    variance = sum_square - float(sum_x.square().sum()) / len(source)
    result = {"method": "training-fitted rank-k PCA reference", "rank": rank, "development_fvu": error / variance, "development_mean_cosine": cosine_sum / len(source), "train_tokens": len(np.load(matrix / "train.npy", mmap_mode="r")), "development_tokens": len(source), "comparison_scope": "same training vectors and number of scalar coefficients; dictionary storage and representation families differ", "completed_at": timestamp()}
    np.savez(destination / "pca.npz", basis=basis.cpu().numpy(), mean=mean.cpu().numpy(), scale=scale)
    write_json(destination / "pca_reference.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--comparison-only", action="store_true")
    parser.add_argument("--tokens-per-patient", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    storage, _, _ = context()
    root = Path(storage["runs"]) / configuration()["suite"]
    runs = [root / name for name in args.runs]
    destination = root / args.name
    destination.mkdir(exist_ok=True)
    if not args.comparison_only:
        for run in runs:
            print("BOOTSTRAP", run.name, bootstrap_predictions(run), flush=True)
        print("PCA", pca_reference(runs[0], destination), flush=True)
    print("SEEDS", compare_seeds(runs, destination, args.tokens_per_patient), flush=True)


if __name__ == "__main__":
    main()

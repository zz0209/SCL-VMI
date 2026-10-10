import argparse
import gc
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .context import read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_campaign import campaign_root, ensure_identity, pause_requested, progress


def checkpoint_pause(root):
    if pause_requested(root, "local"):
        raise SystemExit(75)


def matrix_directory(root, request):
    return root / request.get("matrix_directory", "matrices") / request["model"] / request["site"]


def weights_from_offsets(offsets):
    lengths = np.diff(offsets)
    return np.repeat(1.0 / lengths / len(lengths), lengths)


@torch.inference_mode()
def pca_reference(root, run_id, matrix=None):
    run = root / "runs" / run_id
    request = read_json(run / "request.json")
    matrix = Path(matrix) if matrix else matrix_directory(root, request)
    destination = run / "pca_reference"
    destination.mkdir(exist_ok=True)
    ensure_identity(destination / "request.json", {"matrix_identity": read_json(matrix / "matrix.json")["identity"], "rank": request["k"], "code_sha256": sha256(Path(__file__))})
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    source = np.load(matrix / "train.npy", mmap_mode="r")
    offsets = np.load(matrix / "train_offsets.npy")
    with np.load(matrix / "normalization.npz") as saved:
        mean = torch.tensor(saved["mean"], device="cuda", dtype=torch.float64)
        scale = float(saved["scale"])
    weights = weights_from_offsets(offsets)
    rank = request["k"]
    assert rank <= source.shape[1]
    started = time.monotonic()
    basis_path = destination / "basis.npz"
    if not basis_path.exists():
        covariance = torch.zeros((source.shape[1], source.shape[1]), device="cuda", dtype=torch.float64)
        for start in range(0, len(source), 4096):
            checkpoint_pause(root)
            batch = (torch.tensor(np.array(source[start:start + 4096]), device="cuda", dtype=torch.float64) - mean) / scale
            weight = torch.tensor(weights[start:start + len(batch)], device="cuda")
            covariance += batch.T @ (batch * weight[:, None])
            progress(root, f"pca-fit-{run_id}", "running", min(start + len(batch), len(source)), len(source), started, worker="local")
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -rank:].float()
        np.savez(basis_path, basis=basis.cpu().numpy(), eigenvalues=eigenvalues.cpu().numpy(), mean=mean.cpu().numpy(), scale=scale)
        del covariance, eigenvalues, eigenvectors, batch
    else:
        with np.load(basis_path) as saved:
            basis = torch.tensor(saved["basis"], device="cuda")
    torch.testing.assert_close(basis.T @ basis, torch.eye(rank, device="cuda"), rtol=1e-4, atol=1e-5)
    source = np.load(matrix / "development.npy", mmap_mode="r")
    weights = weights_from_offsets(np.load(matrix / "development_offsets.npy"))
    total = torch.zeros(source.shape[1], device="cuda", dtype=torch.float64)
    square = error = cosine = 0.0
    mean = mean.float()
    for start in range(0, len(source), 4096):
        checkpoint_pause(root)
        raw = torch.tensor(np.array(source[start:start + 4096]), device="cuda")
        batch = (raw - mean) / scale
        recovered = batch @ basis @ basis.T
        weight = torch.tensor(weights[start:start + len(batch)], device="cuda")
        total += (batch.double() * weight[:, None]).sum(0)
        square += float((batch.double().square().sum(1) * weight).sum())
        error += float(((batch - recovered).double().square().sum(1) * weight).sum())
        cosine += float((F.cosine_similarity(raw, recovered * scale + mean, dim=1).double() * weight).sum())
        progress(root, f"pca-evaluate-{run_id}", "running", min(start + len(batch), len(source)), len(source), started, worker="local")
    result = {"state": "completed", "method": "training-fitted patient-weighted rank-k PCA", "rank": rank, "development_fvu": error / (square - float(total.square().sum())), "development_cosine": cosine, "weighting": "equal patient weight and equal saved-position weight within each patient", "comparison_scope": "same data and number of scalar coefficients; dictionary storage and representation families differ", "test_used": False, "completed_at": timestamp()}
    write_json(destination / "result.json", result)
    progress(root, f"pca-evaluate-{run_id}", "completed", len(source), len(source), started, worker="local")
    return result


@torch.inference_mode()
def compare_seeds(root, run_ids, destination, matrix=None, samples_per_patient=16):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    requests = [read_json(root / "runs" / run_id / "request.json") for run_id in run_ids]
    for request in requests[1:]:
        for key in ["model", "site", "dict_size", "k", "matrix_identity"]:
            assert request[key] == requests[0][key]
    assert len({request["seed"] for request in requests}) == len(requests)
    matrix = Path(matrix) if matrix else matrix_directory(root, requests[0])
    hashes = [sha256(root / "runs" / run_id / "dictionary.pt") for run_id in run_ids]
    ensure_identity(destination / "request.json", {"dictionaries": hashes, "samples_per_patient": samples_per_patient, "code_sha256": sha256(Path(__file__)), "matrix_identity": read_json(matrix / "matrix.json")["identity"]})
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    source = np.load(matrix / "development.npy", mmap_mode="r")
    offsets = np.load(matrix / "development_offsets.npy")
    generator = np.random.default_rng(20261009)
    pieces = [generator.choice(np.arange(start, end), min(end - start, samples_per_patient), replace=False) for start, end in zip(offsets[:-1], offsets[1:])]
    indices = np.concatenate(pieces)
    weights = np.concatenate([np.full(len(piece), 1.0 / len(piece) / len(pieces)) for piece in pieces])
    np.savez(destination / "sample.npz", indices=indices, weights=weights, matrix_identity=read_json(matrix / "matrix.json")["identity"])
    tokens = torch.tensor(np.array(source[indices]), device="cuda")
    encoded = []
    valid_features = []
    decoder_paths = []
    coverage = []
    started = time.monotonic()
    for run_id in run_ids:
        checkpoint_pause(root)
        encoded_path = destination / f"{run_id}_activations.npy"
        decoder_path = destination / f"{run_id}_decoder.npy"
        valid_path = destination / f"{run_id}_nonconstant.npy"
        receipt_path = destination / f"{run_id}_encoding.json"
        if not receipt_path.exists():
            dictionary = SpatialDictionary(root / "runs" / run_id / "dictionary.pt")
            shape = (len(tokens), dictionary.request["dict_size"])
            values = np.lib.format.open_memmap(encoded_path, mode="w+", dtype=np.float32, shape=shape)
            for start in range(0, len(tokens), 512):
                checkpoint_pause(root)
                batch = dictionary.encode(tokens[start:start + 512]).cpu().numpy()
                values[start:start + len(batch)] = batch
                progress(root, f"seed-encode-{run_id}", "running", min(start + len(batch), len(tokens)), len(tokens), started, worker="local")
            mean = np.zeros(shape[1], dtype=np.float64)
            square = np.zeros(shape[1], dtype=np.float64)
            frequency = np.zeros(shape[1], dtype=np.float64)
            for start in range(0, len(tokens), 512):
                part = np.asarray(values[start:start + 512], dtype=np.float64)
                weight = weights[start:start + len(part), None]
                mean += (part * weight).sum(0)
                square += (part ** 2 * weight).sum(0)
                frequency += ((part > 0) * weight).sum(0)
            deviation = np.sqrt(np.maximum(square - mean ** 2, 0))
            valid = deviation > 1e-8
            np.save(valid_path, valid)
            for start in range(0, len(tokens), 512):
                part = np.asarray(values[start:start + 512], dtype=np.float64)
                part[:, valid] = (part[:, valid] - mean[valid]) / deviation[valid]
                part[:, ~valid] = 0
                values[start:start + len(part)] = part * np.sqrt(weights[start:start + len(part), None])
            values.flush()
            np.save(decoder_path, F.normalize(dictionary.weights["W_dec"], dim=1).cpu().numpy())
            write_json(receipt_path, {"run_id": run_id, "nonconstant_features": int(valid.sum()), "features": len(valid), "weighted_frequency_quantiles": np.quantile(frequency, [0, .1, .5, .9, 1]).tolist(), "completed_at": timestamp()})
            del dictionary, values
            gc.collect()
            torch.cuda.empty_cache()
        encoded.append(np.load(encoded_path, mmap_mode="r"))
        valid_features.append(np.load(valid_path))
        decoder_paths.append(decoder_path)
        coverage.append(read_json(receipt_path))
    comparisons = []
    for first, second in combinations(range(len(run_ids)), 2):
        checkpoint_pause(root)
        pair = f"s{requests[first]['seed']}_s{requests[second]['seed']}"
        result_path = destination / f"{pair}.json"
        if result_path.exists():
            comparisons.append(read_json(result_path))
            continue
        target = torch.tensor(np.array(encoded[second]), device="cuda")
        source_valid = valid_features[first]
        target_valid = valid_features[second]
        assert source_valid.any() and target_valid.any()
        invalid_target = torch.tensor(~target_valid, device="cuda")
        count = encoded[first].shape[1]
        maxima = np.full(count, -np.inf, dtype=np.float32)
        nearest = np.zeros(count, dtype=np.int64)
        reverse_max = torch.full((target.shape[1],), -torch.inf, device="cuda")
        reverse_index = torch.zeros(target.shape[1], dtype=torch.int64, device="cuda")
        for start in range(0, count, 256):
            checkpoint_pause(root)
            batch = torch.tensor(np.array(encoded[first][:, start:start + 256]), device="cuda")
            correlation = batch.T @ target
            correlation[:, invalid_target] = -torch.inf
            correlation[torch.tensor(~source_valid[start:start + batch.shape[1]], device="cuda")] = -torch.inf
            best, match = correlation.max(1)
            maxima[start:start + len(best)] = best.cpu().numpy()
            nearest[start:start + len(best)] = match.cpu().numpy()
            value, index = correlation.max(0)
            improved = value > reverse_max
            reverse_index[improved] = index[improved] + start
            reverse_max = torch.maximum(reverse_max, value)
            progress(root, f"seed-match-{pair}", "running", min(start + 256, count), count, started, worker="local")
        reverse = reverse_index.cpu().numpy()
        maxima[~source_valid] = np.nan
        nearest[~source_valid] = -1
        reciprocal = np.zeros(count, dtype=bool)
        reciprocal[source_valid] = reverse[nearest[source_valid]] == np.flatnonzero(source_valid)
        first_decoder = np.load(decoder_paths[first], mmap_mode="r")
        second_decoder = np.load(decoder_paths[second], mmap_mode="r")
        decoder_cosine = np.empty(count, dtype=np.float32)
        for start in range(0, count, 256):
            decoder_cosine[start:start + 256] = (first_decoder[start:start + 256] * second_decoder[nearest[start:start + 256]]).sum(1)
        decoder_cosine[~source_valid] = np.nan
        np.savez(destination / f"{pair}.npz", source_index=np.arange(count), target_index=nearest, correlation=maxima, reciprocal=reciprocal, decoder_cosine=decoder_cosine, source_nonconstant=source_valid, target_nonconstant=target_valid)
        result = {"pair": pair, "source_run": run_ids[first], "target_run": run_ids[second], "source_eligible_features": int(source_valid.sum()), "target_eligible_features": int(target_valid.sum()), "source_features": len(source_valid), "target_features": len(target_valid), "denominator": "nonconstant source features; matching considers nonconstant target features", "correlation_quantiles": np.quantile(maxima[source_valid], [0, .1, .5, .9, 1]).tolist(), "reciprocal_fraction": float(reciprocal[source_valid].mean()), "reciprocal_and_correlation_ge_0_8": float(np.mean(reciprocal[source_valid] & (maxima[source_valid] >= .8))), "reciprocal_and_correlation_ge_0_5": float(np.mean(reciprocal[source_valid] & (maxima[source_valid] >= .5))), "completed_at": timestamp()}
        write_json(result_path, result)
        comparisons.append(result)
        del target, correlation, batch
        torch.cuda.empty_cache()
    result = {"state": "completed", "method": "patient-weighted activation Pearson correlation; reciprocal nearest features", "runs": run_ids, "patients": len(pieces), "tokens": len(tokens), "maximum_samples_per_patient": samples_per_patient, "coverage": coverage, "comparisons": comparisons, "test_used": False, "completed_at": timestamp()}
    write_json(destination / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--matrix", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = campaign_root(args.root)
    print("PCA", pca_reference(root, args.runs[0], args.matrix), flush=True)
    if len(args.runs) > 1:
        print("SEED_COMPARISON", compare_seeds(root, args.runs, root / "diagnostics" / args.name, args.matrix), flush=True)

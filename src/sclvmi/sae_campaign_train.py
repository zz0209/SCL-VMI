import argparse
import gc
import importlib.util
import math
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock

from .context import ROOT, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary, activations, export_dictionary, load_matrix
from .sae_campaign import campaign_root, configuration, ensure_identity, pause_requested, training_identity


def make_sae(request, config):
    spec = importlib.util.spec_from_file_location("campaign_batchtopk", ROOT / config["upstream"]["module"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    training = request["training"]
    return module.BatchTopKSAE({
        "seed": request["seed"], "act_size": request["input_dim"],
        "dict_size": request["dict_size"], "device": "cuda", "dtype": torch.float32,
        "input_unit_norm": False, "top_k": request["k"], "l1_coeff": 0.0,
        "top_k_aux": min(training["top_k_aux"], request["dict_size"]),
        "aux_penalty": training["aux_penalty"], "n_batches_to_dead": training["inactive_batches"],
    })


def patient_indices(offsets, count, generator):
    patients = torch.randint(len(offsets) - 1, (count,), generator=generator, device=offsets.device)
    lengths = offsets[patients + 1] - offsets[patients]
    within = (torch.rand(count, generator=generator, device=offsets.device) * lengths).long()
    return offsets[patients] + within


@torch.inference_mode()
def calibrate_patient_balanced(sae, train, offsets, tokens, batch_size):
    generator = torch.Generator(device="cuda").manual_seed(41017)
    cutoffs = []
    for start in range(0, tokens, batch_size):
        batch = train[patient_indices(offsets, min(batch_size, tokens - start), generator)]
        values = F.relu((batch - sae.b_dec) @ sae.W_enc)
        selected = torch.topk(values.flatten(), sae.cfg["top_k"] * len(batch), sorted=False).values
        positive = selected[selected > 0]
        assert len(positive) > 0, "Calibration has no positive activations"
        cutoffs.append(positive.min())
    return float(torch.stack(cutoffs).mean())


@torch.inference_mode()
def evaluate_weighted(sae, data, offsets, threshold, mean, scale, batch_size):
    lengths = offsets[1:] - offsets[:-1]
    weights = torch.repeat_interleave(1.0 / lengths.double(), lengths)
    weights /= len(lengths)
    mean_x = (data.double() * weights[:, None]).sum(0)
    counts = torch.zeros(sae.cfg["dict_size"], device=data.device, dtype=torch.int64)
    frequencies = torch.zeros_like(counts, dtype=torch.float64)
    sums = torch.zeros_like(frequencies)
    maxima = torch.zeros_like(frequencies)
    error = torch.zeros((), device=data.device, dtype=torch.float64)
    variance = torch.zeros_like(error)
    cosine_sum = torch.zeros_like(error)
    l0_mean = torch.zeros_like(error)
    l0_parts = []
    for start in range(0, len(data), batch_size):
        x = data[start:start + batch_size]
        weight = weights[start:start + batch_size]
        encoded = activations(sae, x, threshold)
        recovered = encoded @ sae.W_dec + sae.b_dec
        assert torch.isfinite(recovered).all() and torch.isfinite(encoded).all()
        error += (((recovered - x).double().square().sum(-1)) * weight).sum()
        variance += (((x.double() - mean_x).square().sum(-1)) * weight).sum()
        cosine_sum += (F.cosine_similarity(recovered * scale + mean, x * scale + mean, dim=-1).double() * weight).sum()
        active = encoded > 0
        counts += active.sum(0)
        frequencies += (active * weight[:, None]).sum(0)
        sums += (encoded * weight[:, None]).sum(0)
        maxima = torch.maximum(maxima, encoded.amax(0))
        l0 = active.sum(-1)
        l0_mean += (l0 * weight).sum()
        l0_parts.append(l0)
    l0 = torch.cat(l0_parts).float()
    return {
        "fvu": float(error / variance), "cosine": float(cosine_sum),
        "l0_mean": float(l0_mean), "inactive_fraction": float((counts == 0).float().mean()),
        "l0_token_quantiles": torch.quantile(l0, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=data.device)).cpu().tolist(),
        "tokens": len(data), "patients": len(lengths), "threshold": threshold,
        "weighting": "equal patient weight, equal saved-position weight within patient",
    }, {"counts": counts.cpu().numpy(), "patient_weighted_frequency": frequencies.cpu().numpy(), "activation_sum": sums.cpu().numpy(), "activation_max": maxima.cpu().numpy()}


def save_checkpoint(destination, sae, optimizer, generator, step, history, best_fvu, elapsed, identity):
    temporary = destination / "resume.tmp"
    torch.save({
        "state_dict": sae.state_dict(), "inactive": sae.num_batches_not_active,
        "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "step": step, "history": history, "best_fvu": best_fvu,
        "elapsed_seconds": elapsed, "identity": identity,
    }, temporary)
    temporary.replace(destination / "resume.pt")


def fit_job(root, job, worker, stop_after=None):
    config = configuration()
    root = Path(root)
    destination = root / "runs" / job["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    matrix = root / job.get("matrix_directory", "matrices") / job["model"] / job["site"]
    matrix_info = read_json(matrix / "matrix.json")
    request = training_identity(config, job, matrix_info)
    ensure_identity(destination / "request.json", request)
    if (destination / "result.json").exists():
        return read_json(destination / "result.json")
    if pause_requested(root, worker):
        write_json(destination / "status.json", {"state": "paused", "worker": worker, "updated_at": timestamp()})
        return {"state": "paused"}
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in [Path(__file__), ROOT / "src/sclvmi/sae.py", ROOT / "src/sclvmi/sae_campaign.py", ROOT / config["upstream"]["module"]]}
    invocation = {"started_at": timestamp(), "host": socket.gethostname(), "worker": worker, "pid": os.getpid(), "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(), "sources": source_hashes}
    invocation_path = destination / "invocations" / f"{time.time_ns()}.json"
    write_json(invocation_path, invocation)
    write_json(destination / "status.json", {"state": "loading", "step": 0, "total": request["training"]["steps"], "worker": worker, "updated_at": timestamp()})
    with np.load(matrix / "normalization.npz") as normalization:
        mean = torch.tensor(normalization["mean"], device="cuda")
        scale = float(normalization["scale"])
    train = load_matrix(matrix / "train.npy", mean, scale)
    development = load_matrix(matrix / "development.npy", mean, scale)
    train_offsets = torch.tensor(np.load(matrix / "train_offsets.npy"), device="cuda", dtype=torch.int64)
    dev_offsets = torch.tensor(np.load(matrix / "development_offsets.npy"), device="cuda", dtype=torch.int64)
    sae = make_sae(request, config)
    training = request["training"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]))
    generator = torch.Generator(device="cuda").manual_seed(job["seed"] + 12345)
    history, start_step, best_fvu, previous_elapsed = [], 0, math.inf, 0.0
    if (destination / "resume.pt").exists():
        saved = torch.load(destination / "resume.pt", map_location="cuda", weights_only=False)
        assert saved["identity"] == request
        sae.load_state_dict(saved["state_dict"], strict=True)
        sae.num_batches_not_active.copy_(saved["inactive"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        history, start_step, best_fvu = saved["history"], saved["step"], saved["best_fvu"]
        previous_elapsed = saved["elapsed_seconds"]
        del saved
        print("RESUME", job["run_id"], start_step, flush=True)
    started = time.monotonic()
    last_checkpoint, last_control, last_progress = started, 0.0, 0.0
    total = training["steps"]
    checkpoint_step = start_step
    batch_size = min(training["batch_size"], len(train))
    stopped = False
    for step in range(start_step + 1, total + 1):
        warmup = min(training["warmup_steps"], max(1, total // 10))
        decay_start = int(total * training["decay_start_fraction"])
        factor = min(1.0, step / warmup)
        if step > decay_start:
            factor *= 0.1 + 0.9 * (total - step) / max(1, total - decay_start)
        for group in optimizer.param_groups:
            group["lr"] = training["learning_rate"] * factor
        indices = patient_indices(train_offsets, batch_size, generator)
        output = sae(train[indices])
        loss = output["loss"]
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), training["gradient_clip"], error_if_nonfinite=True)
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        now = time.monotonic()
        if now - last_control >= config["control"]["poll_seconds"] or step == stop_after:
            stopped = pause_requested(root, worker) or step == stop_after
            last_control = now
        if step == 1 or step % training["validate_every"] == 0 or step == total:
            threshold = calibrate_patient_balanced(sae, train, train_offsets, request["inference"]["calibration_train_tokens"], batch_size)
            metrics, _ = evaluate_weighted(sae, development, dev_offsets, threshold, mean, scale, batch_size)
            metrics.update({"step": step, "updated_at": timestamp()})
            history.append(metrics)
            write_json(destination / "history.json", history)
            if metrics["fvu"] < best_fvu:
                best_fvu = metrics["fvu"]
                export_dictionary(sae, request, threshold, mean, scale, destination / "best.pt", step)
            print("VALIDATION", job["run_id"], metrics, flush=True)
        elapsed = previous_elapsed + time.monotonic() - started
        if step % training["checkpoint_every"] == 0 or time.monotonic() - last_checkpoint >= training["checkpoint_seconds"] or step == total or stopped:
            save_checkpoint(destination, sae, optimizer, generator, step, history, best_fvu, elapsed, request)
            checkpoint_step = step
            last_checkpoint = time.monotonic()
        if step == 1 or time.monotonic() - last_progress >= 3 or step == total or stopped:
            status = {
                "state": "paused" if stopped else "running", "run_id": job["run_id"],
                "worker": worker, "step": step, "total": total,
                "loss": float(loss.detach()), "l0": float(output["l0_norm"]),
                "gradient_norm": float(norm), "elapsed_seconds": elapsed,
                "steps_per_second": (step - start_step) / max(time.monotonic() - started, 1e-9),
                "checkpoint_step": checkpoint_step,
                "updated_at": timestamp(),
            }
            write_json(destination / "status.json", status)
            print("TRAIN", job["run_id"], step, total, status["steps_per_second"], flush=True)
            last_progress = time.monotonic()
        del output, loss, indices
        if stopped:
            return {"state": "paused", "step": step}
    best = torch.load(destination / "best.pt", map_location="cuda", weights_only=True)
    sae.load_state_dict(best["state_dict"], strict=True)
    threshold = best["threshold"]
    metrics, statistics = evaluate_weighted(sae, development, dev_offsets, threshold, mean, scale, batch_size)
    np.savez(destination / "development_feature_statistics.npz", **statistics)
    export_dictionary(sae, request, threshold, mean, scale, destination / "dictionary.pt", best["step"])
    loaded = SpatialDictionary(destination / "dictionary.pt")
    actual = development[:127] * scale + mean
    torch.testing.assert_close(loaded.encode(actual), torch.cat([loaded.encode(part) for part in actual.split(17)]), rtol=1e-5, atol=config["inference"]["batch_independence_atol"])
    limits = config["acceptance"]
    checks = {"fvu": metrics["fvu"] <= limits["max_development_fvu"], "cosine": metrics["cosine"] >= limits["min_mean_token_cosine"], "inactive": metrics["inactive_fraction"] <= limits["max_inactive_fraction"], "sparsity": abs(metrics["l0_mean"] / job["k"] - 1) <= limits["l0_relative_tolerance"]}
    result = {"state": "completed", "run_id": job["run_id"], "request": request, "steps": total, "best_step": best["step"], "development": metrics, "feature_checks": checks, "feature_checks_pass": all(checks.values()), "dictionary_sha256": sha256(destination / "dictionary.pt"), "elapsed_seconds": previous_elapsed + time.monotonic() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "completed_at": timestamp(), "test_used": False}
    write_json(destination / "result.json", result)
    write_json(destination / "status.json", {"state": "completed", "worker": worker, "step": total, "total": total, "updated_at": timestamp()})
    invocation.update({"completed_at": timestamp(), "result": "completed", "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
    write_json(invocation_path, invocation)
    del sae, optimizer, loaded, train, development
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    root = campaign_root(args.root)
    job = read_json(args.job)
    lock = root / "runs" / job["run_id"] / "execution.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock), timeout=0):
        result = fit_job(root, job, args.worker, args.stop_after)
    print(result, flush=True)
    if result["state"] == "paused":
        raise SystemExit(75)


if __name__ == "__main__":
    main()

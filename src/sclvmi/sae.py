import argparse
import gc
import importlib.util
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .context import ROOT, context, read_json, sha256, timestamp, write_json


def load_upstream():
    config = read_json(ROOT / "configs/sae_preparation.json")
    path = ROOT / config["upstream"]["module"]
    spec = importlib.util.spec_from_file_location("sclvmi_batchtopk_upstream", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatchTopKSAE


def activations(sae, x, threshold=None):
    values = F.relu((x - sae.b_dec) @ sae.W_enc)
    if sae.cfg.get("sae_type") == "topk":
        selected = torch.topk(values, sae.cfg["top_k"], dim=-1)
        return torch.zeros_like(values).scatter(-1, selected.indices, selected.values)
    if threshold is not None:
        return values * (values > threshold)
    selected = torch.topk(values.flatten(), sae.cfg["top_k"] * len(x), sorted=False)
    return torch.zeros_like(values.flatten()).scatter(0, selected.indices, selected.values).reshape_as(values)


@torch.inference_mode()
def calibrate(sae, train, tokens=65536, batch=2048):
    generator = torch.Generator(device=train.device).manual_seed(41017)
    cutoffs = []
    for start in range(0, min(tokens, len(train)), batch):
        indices = torch.randint(len(train), (min(batch, tokens - start),), generator=generator, device=train.device)
        x = train[indices]
        values = F.relu((x - sae.b_dec) @ sae.W_enc)
        selected = torch.topk(values.flatten(), sae.cfg["top_k"] * len(x), sorted=False).values
        positive = selected[selected > 0]
        assert positive.numel() > 0
        cutoffs.append(positive.min())
    return float(torch.stack(cutoffs).mean())


@torch.inference_mode()
def evaluate(sae, data, threshold, mean, scale, batch=2048):
    dimension = sae.cfg["dict_size"]
    counts = torch.zeros(dimension, device=data.device, dtype=torch.int64)
    sums = torch.zeros(dimension, device=data.device, dtype=torch.float64)
    maxima = torch.zeros(dimension, device=data.device)
    mean_x = data.mean(0)
    error = torch.zeros((), device=data.device, dtype=torch.float64)
    variance = torch.zeros_like(error)
    cosine_sum = torch.zeros_like(error)
    centered_cosine_sum = torch.zeros_like(error)
    l0_values = []
    for start in range(0, len(data), batch):
        x = data[start:start + batch]
        features = activations(sae, x, threshold)
        recovered = features @ sae.W_dec + sae.b_dec
        assert torch.isfinite(recovered).all() and torch.isfinite(features).all()
        error += (recovered - x).double().square().sum()
        variance += (x - mean_x).double().square().sum()
        centered_cosine_sum += F.cosine_similarity(recovered, x, dim=-1).double().sum()
        cosine_sum += F.cosine_similarity(recovered * scale + mean, x * scale + mean, dim=-1).double().sum()
        counts += (features > 0).sum(0)
        sums += features.double().sum(0)
        maxima = torch.maximum(maxima, features.amax(0))
        l0_values.append((features > 0).sum(-1))
    l0 = torch.cat(l0_values).float()
    metrics = {"fvu": float(error / variance), "mse_normalized": float(error / data.numel()), "cosine": float(cosine_sum / len(data)), "centered_cosine": float(centered_cosine_sum / len(data)), "l0_mean": float(l0.mean()), "l0_quantiles": torch.quantile(l0, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=l0.device)).cpu().tolist(), "inactive_fraction": float((counts == 0).float().mean()), "tokens": len(data), "threshold": threshold}
    statistics = {"counts": counts.cpu().numpy(), "activation_sum": sums.cpu().numpy(), "activation_max": maxima.cpu().numpy()}
    return metrics, statistics


def export_dictionary(sae, request, threshold, mean, scale, path, step):
    artifact = {"state_dict": {key: value.detach().cpu() for key, value in sae.state_dict().items()}, "request": request, "threshold": threshold, "mean": mean.detach().cpu(), "scale": float(scale), "step": step, "format_version": 1}
    temporary = Path(path).with_suffix(".tmp")
    torch.save(artifact, temporary)
    temporary.replace(path)


class SpatialDictionary:
    def __init__(self, path, device="cuda"):
        self.path = Path(path)
        self.artifact = torch.load(self.path, map_location=device, weights_only=True)
        self.request = self.artifact["request"]
        self.mean = self.artifact["mean"].to(device)
        self.scale = self.artifact["scale"]
        self.threshold = self.artifact["threshold"]
        self.weights = {key: value.to(device) for key, value in self.artifact["state_dict"].items()}
        self.device = device

    @torch.inference_mode()
    def encode(self, tokens):
        x = (tokens.to(self.device) - self.mean) / self.scale
        values = F.relu((x - self.weights["b_dec"]) @ self.weights["W_enc"])
        if self.request["training"].get("sae_type") == "topk":
            selected = torch.topk(values, self.request["k"], dim=-1)
            return torch.zeros_like(values).scatter(-1, selected.indices, selected.values)
        return values * (values > self.threshold)

    @torch.inference_mode()
    def decode(self, features):
        return (features @ self.weights["W_dec"] + self.weights["b_dec"]) * self.scale + self.mean

    @torch.inference_mode()
    def reconstruct(self, tokens, batch=2048):
        return torch.cat([self.decode(self.encode(tokens[start:start + batch])) for start in range(0, len(tokens), batch)])

    @torch.inference_mode()
    def spatial_activations(self, feature_map, batch=2048):
        assert feature_map.ndim == 4
        channels, *shape = feature_map.shape
        tokens = feature_map.reshape(channels, -1).T
        values = torch.cat([self.encode(tokens[start:start + batch]) for start in range(0, len(tokens), batch)])
        return values.T.reshape(-1, *shape)


def load_matrix(path, mean, scale, device="cuda"):
    source = np.load(path, mmap_mode="r")
    tensor = torch.empty(source.shape, device=device, dtype=torch.float32)
    for start in range(0, len(source), 8192):
        values = torch.from_numpy(np.array(source[start:start + 8192], copy=True)).to(device)
        tensor[start:start + len(values)] = (values - mean) / scale
    return tensor


def fit(name, site, run_id, steps=None, k=32, expansion=4, seed=2025, smoke=False, stop_after=None):
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(ROOT / "configs/sae_preparation.json")
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["suite"]
    destination = root / run_id
    destination.mkdir(parents=True, exist_ok=True)
    matrix = Path(storage["activations"]) / config["suite"] / name / (f"{site}_smoke" if smoke else site)
    matrix_info = read_json(matrix / "matrix.json")
    dimension = matrix_info["channels"]
    training = dict(config["training"])
    training["steps"] = steps or training["steps"]
    steps = training["steps"]
    request = {"run_id": run_id, "model": name, "site": site, "input_dim": dimension, "dict_size": dimension * expansion, "k": k, "expansion": expansion, "seed": seed, "training": training, "inference": config["inference"], "acceptance": config["acceptance"], "preparation_config_sha256": sha256(ROOT / "configs/sae_preparation.json"), "matrix": str(matrix), "matrix_sha256": sha256(matrix / "matrix.json"), "upstream": config["upstream"], "upstream_sha256": sha256(ROOT / config["upstream"]["module"]), "source_sha256": sha256(Path(__file__)), "environment": {"spec_sha256": sha256(ROOT / "configs/environments/pipeline.json"), "resolved_dependencies_sha256": sha256(ROOT / "configs/environments/pipeline.lock.txt"), "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}, "smoke": smoke, "test_used": False}
    if (destination / "request.json").exists():
        assert read_json(destination / "request.json") == request, "Changed request requires a new run ID"
    else:
        write_json(destination / "request.json", request)
    if (destination / "result.json").exists():
        print("REUSE_COMPLETED", run_id, flush=True)
        return read_json(destination / "result.json")
    with np.load(matrix / "normalization.npz") as normalization:
        mean = torch.tensor(normalization["mean"], device="cuda")
        scale = float(normalization["scale"])
    print(f"Loading {name}/{site} patient-balanced matrices", flush=True)
    train = load_matrix(matrix / "train.npy", mean, scale)
    development = load_matrix(matrix / "development.npy", mean, scale)
    sae_cfg = {"seed": seed, "act_size": dimension, "dict_size": dimension * expansion, "device": "cuda", "dtype": torch.float32, "input_unit_norm": False, "top_k": k, "l1_coeff": 0.0, "top_k_aux": training["top_k_aux"], "aux_penalty": training["aux_penalty"], "n_batches_to_dead": training["inactive_batches"]}
    sae = load_upstream()(sae_cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=training["learning_rate"], betas=tuple(training["betas"]))
    generator = torch.Generator(device="cuda").manual_seed(seed + 12345)
    start_step = 0
    best_fvu = math.inf
    history = []
    if (destination / "resume.pt").exists():
        saved = torch.load(destination / "resume.pt", map_location="cuda", weights_only=False)
        sae.load_state_dict(saved["state_dict"], strict=True)
        sae.num_batches_not_active.copy_(saved["inactive"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"].cpu())
        start_step = saved["step"]
        best_fvu = saved["best_fvu"]
        history = saved["history"]
        del saved
        print("RESUME", run_id, start_step, flush=True)
    started = time.perf_counter()
    batch_size = min(training["batch_size"], len(train))
    validate_every = min(training["validate_every"], steps)
    threshold = 0.0
    for step in range(start_step + 1, steps + 1):
        warmup = min(training["warmup_steps"], max(1, steps // 10))
        decay_start = int(steps * training["decay_start_fraction"])
        factor = min(1.0, step / warmup)
        if step > decay_start:
            factor *= 0.1 + 0.9 * (steps - step) / max(1, steps - decay_start)
        for group in optimizer.param_groups:
            group["lr"] = training["learning_rate"] * factor
        indices = torch.randint(len(train), (batch_size,), generator=generator, device="cuda")
        batch = train[indices]
        output = sae(batch)
        loss = output["loss"]
        assert torch.isfinite(loss), f"Non-finite loss at step {step}"
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(sae.parameters(), training["gradient_clip"], error_if_nonfinite=True)
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        del batch, indices
        if step == 1 or step % 100 == 0 or step == steps:
            elapsed = time.perf_counter() - started
            progress = {"run_id": run_id, "stage": "train", "step": step, "total": steps, "loss": float(loss.detach()), "l0": float(output["l0_norm"]), "aux_loss": float(output["aux_loss"].detach()), "gradient_norm": float(norm), "elapsed_seconds": elapsed, "steps_per_second": (step - start_step) / max(elapsed, 1e-9), "updated_at": timestamp()}
            write_json(root / "training_progress.json", progress)
            print(f"{run_id}: {step}/{steps} loss={progress['loss']:.5f} L0={progress['l0']:.2f} {progress['steps_per_second']:.1f} steps/s", flush=True)
        if step == 1 or step % validate_every == 0 or step == steps:
            threshold = calibrate(sae, train, tokens=config["inference"]["calibration_train_tokens"], batch=batch_size)
            metrics, _ = evaluate(sae, development, threshold, mean, scale, batch=batch_size)
            metrics.update({"step": step, "updated_at": timestamp()})
            history.append(metrics)
            write_json(destination / "history.json", history)
            if metrics["fvu"] < best_fvu:
                best_fvu = metrics["fvu"]
                export_dictionary(sae, request, threshold, mean, scale, destination / "best.pt", step)
            print(f"VALIDATION {run_id}: FVU={metrics['fvu']:.5f} cosine={metrics['cosine']:.5f} L0={metrics['l0_mean']:.2f} inactive={metrics['inactive_fraction']:.3%}", flush=True)
        if step % training["checkpoint_every"] == 0 or step == steps or step == stop_after:
            temporary = destination / "resume.tmp"
            torch.save({"state_dict": sae.state_dict(), "inactive": sae.num_batches_not_active, "optimizer": optimizer.state_dict(), "generator": generator.get_state(), "step": step, "best_fvu": best_fvu, "history": history}, temporary)
            temporary.replace(destination / "resume.pt")
            write_json(destination / "status.json", {"status": "paused" if step == stop_after else "running", "step": step, "total": steps, "updated_at": timestamp()})
        if step == stop_after:
            print("CHECKPOINT_STOP", run_id, step, flush=True)
            return {"status": "paused", "step": step}
        del output, loss
    best = torch.load(destination / "best.pt", map_location="cuda", weights_only=True)
    sae.load_state_dict(best["state_dict"], strict=True)
    threshold = best["threshold"]
    metrics, statistics = evaluate(sae, development, threshold, mean, scale, batch=batch_size)
    train_metrics, _ = evaluate(sae, train[:min(len(train), 65536)], threshold, mean, scale, batch=batch_size)
    np.savez(destination / "development_feature_statistics.npz", **statistics)
    export_dictionary(sae, request, threshold, mean, scale, destination / "dictionary.pt", best["step"])
    loaded = SpatialDictionary(destination / "dictionary.pt")
    actual_tokens = development[:127] * scale + mean
    encoded = loaded.encode(actual_tokens)
    torch.testing.assert_close(encoded, torch.cat([loaded.encode(part) for part in actual_tokens.split(17)]), rtol=1e-5, atol=config["inference"]["batch_independence_atol"])
    reconstructed = loaded.reconstruct(actual_tokens)
    manual = activations(sae, development[:127], threshold) @ sae.W_dec + sae.b_dec
    torch.testing.assert_close(reconstructed, manual * scale + mean, rtol=1e-5, atol=1e-5)
    assert all(torch.isfinite(value).all() for value in loaded.weights.values())
    gates = config["acceptance"]
    feature_checks = {"fvu": metrics["fvu"] <= gates["max_development_fvu"], "cosine": metrics["cosine"] >= gates["min_mean_token_cosine"], "inactive": metrics["inactive_fraction"] <= gates["max_inactive_fraction"], "sparsity": abs(metrics["l0_mean"] / k - 1) <= gates["l0_relative_tolerance"]}
    result = {"status": "completed", "run_id": run_id, "model": name, "site": site, "seed": seed, "k": k, "dict_size": dimension * expansion, "steps": steps, "best_step": best["step"], "development": metrics, "training_sample": train_metrics, "feature_checks": feature_checks, "feature_checks_pass": all(feature_checks.values()), "downstream_evaluation": "pending", "dictionary_sha256": sha256(destination / "dictionary.pt"), "reload_and_batch_independence": "passed", "elapsed_seconds_this_invocation": time.perf_counter() - started, "completed_at": timestamp(), "smoke": smoke, "test_used": False}
    write_json(destination / "result.json", result)
    write_json(destination / "status.json", {"status": "completed", "step": steps, "updated_at": timestamp()})
    print("SAE_COMPLETED", run_id, metrics, flush=True)
    del train, development, sae, optimizer, loaded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vista", "fmcib"], required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    fit(args.model, args.site, args.run_id, args.steps, args.k, args.expansion, args.seed, args.smoke, args.stop_after)


if __name__ == "__main__":
    main()

import gc
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from eva.vision.models.networks.abmil import ABMIL
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from tqdm import tqdm

from .context import context, read_json, sha256, timestamp, write_json
from .frozen_features import SUITE
from .head import metrics


def load_features(directory, limit=None):
    frames, bags, embeddings = [], [], []
    for split in ["train", "development"]:
        frame = pd.read_csv(directory / f"{split}_members.csv", dtype={"PatientID": str})
        if limit:
            frame = frame.groupby("label", group_keys=False).head(limit // 2).sort_values("identifier")
        tokens, vectors = [], []
        for identifier in tqdm(frame.identifier, desc=f"Load {split}", mininterval=3):
            with np.load(directory / f"{identifier}.npz") as data:
                tokens.append(data["tokens"])
                vectors.append(data["embedding"])
        frames.append(frame.reset_index(drop=True))
        bags.append(np.stack(tokens))
        embeddings.append(np.stack(vectors))
    assert set(frames[0].PatientID).isdisjoint(frames[1].PatientID)
    return frames, bags, embeddings


class SpatialHead(nn.Module):
    def __init__(self, channels, mode, mean, scale):
        super().__init__()
        self.mode = mode
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.network = ABMIL(input_size=channels, output_size=1, projected_input_size=128, hidden_size_attention=128, hidden_sizes_mlp=(64,), dropout_input_embeddings=0.1, dropout_attention=0.0, dropout_mlp=0.1, pad_value=None)
        if mode != "attention":
            self.network.gated_attention.requires_grad_(False)

    def forward(self, x):
        x = (x - self.mean) / self.scale
        if self.mode == "attention":
            return self.network(x).flatten()
        x = self.network.projector(x)
        x = x.amax(1) if self.mode == "max" else x.mean(1)
        return self.network.classifier(x).flatten()


def probabilities(head, x, batch=64):
    head.eval()
    with torch.inference_mode():
        return np.concatenate([head(x[start:start + batch].cuda()).sigmoid().cpu().numpy() for start in range(0, len(x), batch)])


def fit_linear(model_name, directory, frames, embeddings, root, arm="native_linear"):
    output = root / f"{model_name}_{arm}"
    if (output / "result.json").exists():
        result = read_json(output / "result.json")
        assert Path(result["feature_directory"]) == directory, "Use a distinct variant for changed features"
        return result
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trials, models = [], []
    with threadpool_limits(limits=4):
        for value in tqdm([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10], desc=f"{model_name} logistic C"):
            write_json(root / "progress.json", {"status": "running", "stage": "linear_head", "model": model_name, "run_id": output.name, "completed": len(trials), "total": 8, "C": value, "updated_at": timestamp()})
            head = make_pipeline(StandardScaler(), LogisticRegression(C=value, max_iter=3000, random_state=2025))
            head.fit(embeddings[0], frames[0].label)
            trials.append({"C": value, **metrics(frames[1].label, head.predict_proba(embeddings[1])[:, 1])})
            models.append(head)
    index = max(range(len(trials)), key=lambda i: (trials[i]["auroc"], -trials[i]["log_loss"]))
    head = models[index]
    joblib.dump(head, output / "head.joblib")
    p = head.predict_proba(embeddings[1])[:, 1]
    np.testing.assert_array_equal(p, joblib.load(output / "head.joblib").predict_proba(embeddings[1])[:, 1])
    frames[1].assign(probability=p).to_csv(output / "development_predictions.csv", index=False)
    result = {"model": model_name, "arm": arm, "run_id": output.name, "status": "completed", "metrics": trials[index], "trials": trials, "seconds": time.perf_counter() - started, "feature_directory": str(directory), "completed_at": timestamp(), "reload": "passed", "test_used": False}
    write_json(output / "result.json", result)
    return result


def fit_spatial(model_name, mode, directory, frames, bags, root, candidate, seed, lr, epochs=60, positive_weight=1.0):
    output = root / f"{model_name}_{mode}_c{candidate}_s{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        result = read_json(output / "result.json")
        assert Path(result["feature_directory"]) == directory, "Use a distinct variant for changed features"
        assert result["lr"] == lr and result["epochs"] == epochs and result.get("positive_weight", 1.0) == positive_weight, "Use a distinct variant for changed training settings"
        return result
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    request = {"model": model_name, "arm": mode, "candidate": candidate, "seed": seed, "lr": lr, "weight_decay": 1e-3, "batch": 64, "epochs": epochs, "patience": 10, "positive_weight": positive_weight, "feature_config_sha256": sha256(directory / "config.json"), "source": sha256(Path(__file__)), "train_cases": len(frames[0]), "development_cases": len(frames[1])}
    if (output / "request.json").exists():
        assert read_json(output / "request.json") == request
    write_json(output / "request.json", request)
    mean = bags[0].mean(axis=(0, 1))
    scale = np.sqrt(np.maximum((bags[0] * bags[0]).mean(axis=(0, 1)) - mean * mean, 1e-8))
    head = SpatialHead(bags[0].shape[-1], mode, mean, scale).cuda()
    optimizer = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    train_x, dev_x = [torch.from_numpy(x).cuda() for x in bags]
    labels = torch.tensor(frames[0].label.to_numpy(), dtype=torch.float32)
    loss_weight = torch.tensor(positive_weight, dtype=torch.float32, device="cuda")
    history, best, best_key, stale, start_epoch, elapsed = [], None, (-1, -float("inf")), 0, 0, 0
    if (output / "resume.pt").exists():
        saved = torch.load(output / "resume.pt", weights_only=False, map_location="cpu")
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_key, stale = saved["history"], saved["best"], tuple(saved["best_key"]), saved["stale"]
        start_epoch, elapsed = saved["epoch"], saved["seconds"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
    started = time.perf_counter()
    for epoch in tqdm(range(start_epoch, epochs), desc=output.name, mininterval=2):
        if stale >= 10:
            break
        head.train()
        order = torch.randperm(len(train_x))
        losses = []
        for index in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_x[index].cuda())
            loss = F_binary(logits, labels[index].cuda(), pos_weight=loss_weight)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        p = probabilities(head, dev_x)
        score = metrics(frames[1].label, p)
        key = (score["auroc"], -score["log_loss"])
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **score})
        if key > best_key:
            best_key, stale = key, 0
            best = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            torch.save(best, output / "best.pt")
        else:
            stale += 1
        seconds = elapsed + time.perf_counter() - started
        saved = {"head": head.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best": best, "best_key": best_key, "stale": stale, "epoch": epoch + 1, "seconds": seconds, "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
        torch.save(saved, output / "resume.tmp")
        (output / "resume.tmp").replace(output / "resume.pt")
        write_json(output / "history.json", history)
        write_json(root / "progress.json", {"status": "running", "stage": "head_training", "model": model_name, "run_id": output.name, "epoch": epoch + 1, "total": epochs, "seconds": seconds, "metrics": score, "updated_at": timestamp()})
    assert best is not None
    head.load_state_dict(best)
    p = probabilities(head, dev_x)
    reloaded = SpatialHead(bags[0].shape[-1], mode, mean, scale).cuda()
    reloaded.load_state_dict(torch.load(output / "best.pt", map_location="cpu", weights_only=True))
    np.testing.assert_array_equal(p, probabilities(reloaded, dev_x))
    frames[1].assign(probability=p).to_csv(output / "development_predictions.csv", index=False)
    result = {**request, "run_id": output.name, "status": "completed", "metrics": metrics(frames[1].label, p), "selected_epoch": max(history, key=lambda r: (r["auroc"], -r["log_loss"]))["epoch"], "seconds": elapsed + time.perf_counter() - started, "feature_directory": str(directory), "parameters": sum(p.numel() for p in head.parameters() if p.requires_grad), "reload": "passed", "test_used": False, "completed_at": timestamp()}
    write_json(output / "result.json", result)
    del head, reloaded, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


F_binary = nn.functional.binary_cross_entropy_with_logits


def run_heads(name, directory, smoke=False, variant=None):
    storage, _, _ = context()
    suite = SUITE if variant is None else SUITE + "_" + variant
    root = Path(storage["runs"]) / (suite + "_smoke_" + Path(directory).name if smoke else suite)
    root.mkdir(parents=True, exist_ok=True)
    frames, bags, embeddings = load_features(Path(directory), 16 if smoke else None)
    fit_linear(name, Path(directory), frames, embeddings, root)
    from .spatial_linear import spatial_vector

    fit_linear(name, Path(directory), frames, [spatial_vector(bag, name) for bag in bags], root, arm="spatial_linear")
    for mode in ["native_mlp", "mean", "max", "attention"]:
        inputs = [x[:, None, :] for x in embeddings] if mode == "native_mlp" else bags
        candidates = [fit_spatial(name, mode, Path(directory), frames, inputs, root, i, 2025, lr, epochs=2 if smoke else 60) for i, lr in enumerate([1e-4, 3e-4] if not smoke else [3e-4])]
        selected = max(candidates, key=lambda r: (r["metrics"]["auroc"], -r["metrics"]["log_loss"]))
        if not smoke:
            for seed in [2026, 2027]:
                fit_spatial(name, mode, Path(directory), frames, inputs, root, selected["candidate"], seed, selected["lr"])
    return root

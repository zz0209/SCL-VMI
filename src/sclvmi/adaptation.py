import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .head import feature_table, metrics
from .models import load_encoder, spatial_features
from .preprocessing import preprocess


def setup(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def last_block(model, name):
    return {"fmcib": lambda: model.layer4,
            "ctfm": lambda: model.layers[-1],
            "vista": lambda: model.image_encoder.encoder.layers[-1],
            "genesis": lambda: model.down_tr512}[name]()


class Classifier(nn.Module):
    def __init__(self, mean, scale, kind="linear", hidden=256, dropout=0.1):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        dimension = len(mean)
        self.layers = nn.Linear(dimension, 1) if kind == "linear" else nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.layers((features-self.mean)/self.scale).flatten()


def translated(image, spacing, maximum_mm):
    theta = torch.eye(3, 4, device=image.device).unsqueeze(0).repeat(len(image), 1, 1)
    extent = torch.tensor(image.shape[2:], device=image.device) * torch.tensor(spacing, device=image.device)
    shift = (torch.rand(len(image), 3, device=image.device)*2-1)*maximum_mm
    theta[:, :, 3] = (2*shift/extent).flip(1)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def tensor_cache(name, feature_directory, tables):
    storage, _, _ = context()
    signature = read_json(feature_directory / "config.json")
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(storage["preprocessed"]) / "adaptation_inputs" / name / identity
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", signature)
    members = pd.concat(tables).drop_duplicates("identifier")
    for row in tqdm(members.itertuples(), total=len(members), desc=f"{name} input cache", mininterval=2):
        path = directory / f"{row.identifier}.npy"
        if not path.exists():
            value = preprocess(row.image, name).numpy()
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, value)
            temporary.replace(path)
    return directory


class Images(Dataset):
    def __init__(self, table, directory):
        self.identifiers = table.identifier.tolist()
        self.labels = table.label.to_numpy(dtype=np.float32)
        self.directory = directory

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, index):
        image = np.load(self.directory / (self.identifiers[index]+".npy"))
        return torch.from_numpy(image), torch.tensor(self.labels[index])


def predict_batches(encoder, name, head, loader):
    head.eval()
    if encoder is not None:
        encoder.eval()
    probabilities = []
    with torch.inference_mode():
        for inputs, _ in loader:
            inputs = inputs.cuda()
            features = inputs if encoder is None else spatial_features(encoder, name, inputs).mean((2, 3, 4))
            probabilities.extend(head(features).sigmoid().cpu().numpy().tolist())
    return np.asarray(probabilities)


def save_weights(path, encoder, name, head):
    payload = {"head": head.state_dict(), "block": None if encoder is None else last_block(encoder, name).state_dict()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def fit(name, baseline, output, recipe, train, dev, train_x, dev_x, image_directory=None):
    output.mkdir(parents=True, exist_ok=True)
    request = {"model": name, "baseline": str(baseline), "recipe": recipe,
               "feature_config_sha256": sha256(Path(read_json(baseline / "result.json")["feature_directory"]) / "config.json"),
               "code": {file: sha256(ROOT / "src/sclvmi" / file) for file in ("adaptation.py", "models.py", "preprocessing.py")}}
    request_path = output / "request.json"
    if request_path.exists():
        assert read_json(request_path) == request, "Changed experiment requires a new run ID"
    else:
        write_json(request_path, request)
    if (output / "result.json").exists():
        return read_json(output / "result.json")
    setup(recipe["seed"])
    baseline_head = joblib.load(baseline / "head.joblib")
    scaler = baseline_head[0] if recipe["mode"] == "last_block" else StandardScaler().fit(train_x)
    head = Classifier(scaler.mean_, scaler.scale_, recipe["head"], recipe.get("hidden", 256), recipe.get("dropout", .1)).cuda()
    encoder = None
    if recipe["mode"] == "last_block":
        encoder, _ = load_encoder(name)
        encoder = encoder.cuda()
        block = last_block(encoder, name)
        block.requires_grad_(True)
        with torch.no_grad():
            head.layers.weight.copy_(torch.as_tensor(baseline_head[1].coef_, device="cuda", dtype=torch.float32))
            head.layers.bias.copy_(torch.as_tensor(baseline_head[1].intercept_, device="cuda", dtype=torch.float32))
        groups = [{"params": block.parameters(), "lr": recipe["lr"]}, {"params": head.parameters(), "lr": recipe["lr"]*recipe["head_lr_multiplier"]}]
        train_data, dev_data = Images(train, image_directory), Images(dev, image_directory)
    else:
        groups = [{"params": head.parameters(), "lr": recipe["lr"]}]
        train_data = TensorDataset(torch.from_numpy(train_x).float(), torch.from_numpy(train.label.to_numpy()).float())
        dev_data = TensorDataset(torch.from_numpy(dev_x).float(), torch.from_numpy(dev.label.to_numpy()).float())
    optimizer = torch.optim.AdamW(groups, weight_decay=recipe["weight_decay"])
    dev_loader = DataLoader(dev_data, batch_size=recipe["batch_size"], shuffle=False, num_workers=0)
    start, history, elapsed = 0, [], 0.0
    resume = output / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        head.load_state_dict(state["head"])
        if encoder is not None:
            last_block(encoder, name).load_state_dict(state["block"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, history, elapsed = state["epoch"], state["history"], state["seconds"]
    train.to_csv(output / "train_members.csv", index=False)
    started = time.perf_counter()
    initial_head = {key: value.detach().clone() for key, value in head.state_dict().items()}
    frozen_witness = None
    if encoder is not None:
        frozen_name, frozen_value = next((key, value) for key, value in encoder.named_parameters() if not value.requires_grad)
        frozen_witness = (frozen_name, frozen_value.detach().clone())
        block_before = {key: value.detach().clone() for key, value in last_block(encoder, name).named_parameters()}
    effective = recipe.get("effective_batch_size", recipe["batch_size"])
    accumulation = effective // recipe["batch_size"]
    assert effective % recipe["batch_size"] == 0
    _, configuration, _ = context()
    for epoch in range(start, recipe["epochs"]):
        if encoder is not None:
            encoder.eval()
        head.train()
        generator = torch.Generator().manual_seed(recipe["seed"]+epoch)
        loader = DataLoader(train_data, batch_size=recipe["batch_size"], shuffle=True, generator=generator, num_workers=0)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (inputs, labels) in enumerate(tqdm(loader, desc=f"{name} {recipe['arm']} epoch {epoch+1}/{recipe['epochs']}", mininterval=2)):
            inputs, labels = inputs.cuda(), labels.cuda()
            if recipe.get("translation_mm", 0):
                inputs = translated(inputs, configuration["models"][name]["spacing_mm"], recipe["translation_mm"])
            features = inputs if encoder is None else spatial_features(encoder, name, inputs).mean((2, 3, 4))
            loss = F.binary_cross_entropy_with_logits(head(features), labels)
            assert torch.isfinite(loss), "Non-finite training loss"
            group_start = (step//accumulation)*accumulation*recipe["batch_size"]
            group_cases = min(effective, len(train_data)-group_start)
            (loss*len(labels)/group_cases).backward()
            if (step+1) % accumulation == 0 or step+1 == len(loader):
                nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if step % 20 == 0 or step+1 == len(loader):
                write_json(output / "progress.json", {"status": "running", "epoch": epoch+1, "epochs": recipe["epochs"], "batch": step+1, "batches": len(loader), "updated_at": timestamp()})
        probability = predict_batches(encoder, name, head, dev_loader)
        score = metrics(dev.label, probability)
        previous = max(((item["auroc"], -item["log_loss"]) for item in history), default=(-1, -float("inf")))
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), **score})
        if (score["auroc"], -score["log_loss"]) > previous:
            save_weights(output / "best.pt", encoder, name, head)
            dev.assign(probability=probability).to_csv(output / "development_predictions.csv", index=False)
        state = {"head": head.state_dict(), "block": None if encoder is None else last_block(encoder, name).state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1, "history": history, "seconds": elapsed+time.perf_counter()-started, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(state, output / "resume.tmp")
        (output / "resume.tmp").replace(resume)
        write_json(output / "history.json", history)
        print(json.dumps(history[-1]), flush=True)
        best_epoch = max(history, key=lambda item: (item["auroc"], -item["log_loss"]))["epoch"]
        if epoch+1-best_epoch >= recipe.get("patience", recipe["epochs"]):
            break
    if start < len(history):
        assert any(not torch.equal(initial_head[key], value) for key, value in head.state_dict().items()), "Head did not update"
    if frozen_witness is not None:
        assert torch.equal(dict(encoder.named_parameters())[frozen_witness[0]], frozen_witness[1]), "Frozen encoder parameters changed"
        if start < len(history):
            assert any(not torch.equal(block_before[key], value) for key, value in last_block(encoder, name).named_parameters()), "Encoder block did not update"
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["head"])
    if encoder is not None:
        last_block(encoder, name).load_state_dict(checkpoint["block"])
    repeated = predict_batches(encoder, name, head, dev_loader)
    saved = pd.read_csv(output / "development_predictions.csv")
    np.testing.assert_allclose(repeated, saved.probability, rtol=1e-5, atol=1e-6)
    selected = max(history, key=lambda item: (item["auroc"], -item["log_loss"]))
    result = {"status": "completed", "completed_at": timestamp(), "model": name, "recipe": recipe, "selected": selected, "seconds": elapsed+time.perf_counter()-started, "train_cases": len(train), "development_cases": len(dev), "trainable_parameters": sum(p.numel() for group in optimizer.param_groups for p in group["params"]), "checkpoint_reload": "passed", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "test_evaluated": False}
    write_json(output / "result.json", result)
    write_json(output / "progress.json", {"status": "completed", "epoch": len(history), "epochs": recipe["epochs"], "updated_at": timestamp()})
    del head, encoder, optimizer, state, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def load_data(baseline, limit=None):
    result = read_json(baseline / "result.json")
    directory = Path(result["feature_directory"])
    train, train_x = feature_table(directory, "train")
    dev, dev_x = feature_table(directory, "development")
    assert set(train.PatientID).isdisjoint(dev.PatientID)
    if limit:
        def subset(table, values):
            indices = np.concatenate([np.flatnonzero(table.label.to_numpy()==value)[:limit//2] for value in (0, 1)])
            return table.iloc[indices].reset_index(drop=True), values[indices]
        train, train_x = subset(train, train_x)
        dev, dev_x = subset(dev, dev_x)
    return directory, train, dev, train_x, dev_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fmcib", "ctfm", "vista", "genesis"], required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--recipe", required=True)
    args = parser.parse_args()
    storage, _, _ = context()
    baseline = Path(storage["runs"]) / args.baseline
    recipe = read_json(args.recipe)
    directory, train, dev, train_x, dev_x = load_data(baseline)
    images = tensor_cache(args.model, directory, [train, dev]) if recipe["mode"] == "last_block" else None
    fit(args.model, baseline, Path(storage["runs"]) / args.run_id, recipe, train, dev, train_x, dev_x, images)


if __name__ == "__main__":
    main()

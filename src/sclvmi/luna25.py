import argparse
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .head import metrics
from .models import source_module


def configure(directory, batch_size, epochs):
    module = types.ModuleType("experiment_config")
    module.config = SimpleNamespace(MODEL_RGB_I3D=ROOT / "third_party/luna25/resources/model_rgb.pth", PATCH_SIZE=[64, 128, 128], MODE="3D", SEED=2025, NUM_WORKERS=0, SIZE_MM=50, SIZE_PX=64, BATCH_SIZE=batch_size, ROTATION=((-20, 20), (-20, 20), (-20, 20)), TRANSLATION=True, EPOCHS=epochs, PATIENCE=20, LEARNING_RATE=1e-4, WEIGHT_DECAY=5e-4, DATADIR=directory)
    sys.modules["experiment_config"] = module
    return module.config


def prepare(limit):
    storage, _, _ = context()
    destination = Path(storage["preprocessed"]) / "luna25" / "official_npy"
    for name in ["image", "metadata"]:
        (destination / name).mkdir(parents=True, exist_ok=True)
    utilities = source_module("luna25_conversion", ROOT / "third_party/luna25/preprocessing/utils.py")
    table = load_manifest()
    members = {}
    for split in ["train", "development"]:
        members[split] = select_cases(table, split, limit).rename(columns={"identifier": "AnnotationID"})
        for row in tqdm(members[split].itertuples(), total=len(members[split]), desc=f"I3D convert {split}", mininterval=2):
            image_path = destination / "image" / f"{row.AnnotationID}.npy"
            metadata_path = destination / "metadata" / f"{row.AnnotationID}.npy"
            if image_path.exists() and metadata_path.exists():
                continue
            image, metadata = utilities.itk_image_to_numpy_image(sitk.ReadImage(row.image))
            assert image.shape == (64, 128, 128), image.shape
            np.save(image_path, image)
            np.save(metadata_path, metadata)
    return destination, members


def train(run_id, limit=None, epochs=10, batch_size=32, resume=False):
    storage, _, _ = context()
    output = Path(storage["runs"]) / run_id
    output.mkdir(parents=True, exist_ok=resume)
    request = {"limit": limit, "epochs": epochs, "batch_size": batch_size, "dataset_revision": context()[1]["dataset_revision"]}
    request_path = output / "training_request.json"
    if request_path.exists():
        assert read_json(request_path) == request, "Resume requires the original data and configuration"
    write_json(request_path, request)
    if resume and (output / "config.json").exists():
        previous = read_json(output / "config.json")
        assert previous["limit"] == limit and previous["parameters"]["BATCH_SIZE"] == batch_size
        assert previous["parameters"]["EPOCHS"] == epochs, "Resume requires the original training configuration"
    directory, members = prepare(limit)
    config = configure(directory, batch_size, epochs)
    loader_module = source_module("luna25_dataloader", ROOT / "third_party/luna25/dataloader.py")
    model_module = source_module("luna25_model", ROOT / "third_party/luna25/models/model_3d.py")
    torch.set_num_threads(4)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model_module.I3D(num_classes=1, input_channels=3, pre_trained=True, freeze_bn=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_table = members["train"]
    counts = train_table.label.value_counts()
    weights = torch.tensor([len(train_table) / counts[label] for label in train_table.label], dtype=torch.double)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_table))
    train_loader = loader_module.get_data_loader(directory, train_table, mode="3D", sampler=sampler, workers=0, batch_size=batch_size, size_px=64, size_mm=50, rotations=config.ROTATION, translations=True)
    dev_dataset = loader_module.CTCaseDataset(directory, members["development"], size_px=64, size_mm=50, mode="3D")
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    start = 0
    best = -1.0
    history = []
    if resume and (output / "resume.pt").exists():
        state = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start, best, history = state["epoch"], state["best"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
    else:
        assert not (output / "history.json").exists(), "Training history exists without a recovery checkpoint"
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
    write_json(output / "config.json", {"source_commit": "eff27763470640059423a90c05dd018166ea0815", "parameters": configuration, "limit": limit, "purpose": "real-data smoke" if limit else "training", "weight_sha256": sha256(config.MODEL_RGB_I3D), "precision": "float32; TF32 disabled; cuDNN benchmark disabled", "modifications": ["FLARE patient split", "PyTorch 2.8 CUDA 12.8", "serial data loading", "ordered development loader", "resumable epoch checkpoints", "flatten BCE inputs for singleton batches", "explicit float32 convolution for batch-consistent inference"]})
    for split, table in members.items():
        table.to_csv(output / f"{split}_members.csv", index=False)
    for epoch in range(start, epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"I3D epoch {epoch + 1}/{epochs}", mininterval=2):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"].cuda()).reshape(-1)
            loss = criterion(logits, batch["label"].cuda().float().reshape(-1))
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        probabilities = []
        with torch.inference_mode():
            for batch in dev_loader:
                probabilities.extend(model(batch["image"].cuda()).sigmoid().cpu().reshape(-1).tolist())
        score = metrics(members["development"].label, probabilities)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **score})
        if score["auroc"] > best:
            best = score["auroc"]
            torch.save(model.state_dict(), output / "best.pt")
        members["development"].assign(probability=probabilities).to_csv(output / f"development_epoch_{epoch + 1:03d}.csv", index=False)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1, "best": best, "history": history, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
        torch.save(state, output / "resume.tmp")
        (output / "resume.tmp").replace(output / "resume.pt")
        write_json(output / "history.json", history)
        print(history[-1], flush=True)
    model.load_state_dict(torch.load(output / "best.pt", map_location="cuda", weights_only=True), strict=True)
    batch = next(iter(dev_loader))["image"].cuda()
    with torch.inference_mode():
        assert torch.isfinite(model(batch)).all()
    write_json(output / "status.json", {"status": "completed", "completed_at": timestamp(), "epochs": epochs, "best_development_auc": best, "test_evaluated": False, "checkpoint_reload": "passed"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.run_id, args.limit, args.epochs, args.batch_size, args.resume)

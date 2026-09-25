import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .context import context, read_json, timestamp, write_json
from .data import load_manifest, select_cases


def paths(run_id):
    storage, _, _ = context()
    root = Path(storage["runs"]) / run_id
    for key, suffix in [("nnUNet_raw", "raw"), ("nnUNet_preprocessed", "preprocessed"), ("nnUNet_results", "results")]:
        os.environ[key] = str(root / suffix)
        (root / suffix).mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_n_proc_DA"] = "0"
    os.environ["WANDB_DIR"] = str(root)
    os.environ["MPLBACKEND"] = "Agg"
    return root


def prepare(run_id, limit):
    root = paths(run_id)
    request = {"limit": limit, "dataset_revision": context()[1]["dataset_revision"]}
    request_path = root / "preparation_request.json"
    if request_path.exists():
        assert read_json(request_path) == request, "Use a new run ID for different data"
    if (root / "preparation.json").exists():
        assert read_json(root / "preparation.json")["limit"] == limit, "Use a new run ID for different data"
        return
    write_json(request_path, request)
    os.environ["nnUNet_n_proc_DA"] = "2"
    _, _, source = context()
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset, plan_experiment_dataset, preprocess_dataset
    name = "Dataset905_LUNA25"
    raw = root / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    table = load_manifest()
    train = select_cases(table, "train", limit)
    dev = select_cases(table, "development", limit)
    dataset = read_json(source / "dataset.json")
    dataset["name"] = name
    fingerprint = root / "preprocessed" / name / "dataset_fingerprint.json"
    for split, selected in [("train", train), ("development", dev)]:
        for row in tqdm(selected.itertuples(), total=len(selected), desc=f"AutoMSC stage {split}", mininterval=2):
            for column, directory in [("image", "imagesTr"), ("mask", "labelsTr")]:
                target = raw / directory / Path(getattr(row, column)).name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copy2(getattr(row, column), target)
        if split == "train" and not fingerprint.exists():
            dataset["numTraining"] = len(train)
            write_json(raw / "dataset.json", dataset)
            extract_fingerprint_dataset(905, num_processes=1, check_dataset_integrity=True, clean=False, verbose=False)
        if split == "train" and not (root / "preprocessed" / name / "nnUNetPlans.json").exists():
            plan_experiment_dataset(905, gpu_memory_target_in_gb=8)
    dataset["numTraining"] = len(train) + len(dev)
    write_json(raw / "dataset.json", dataset)
    destination = root / "preprocessed" / name
    splits = [{"train": train.identifier.tolist(), "val": dev.identifier.tolist()}]
    write_json(destination / "splits_final.json", splits)
    pd.concat([train, dev])[["identifier", "label"]].to_csv(destination / "cls_data.csv", index=False)
    preprocess_dataset(905, configurations=("3d_fullres",), num_processes=(1,), verbose=False)
    write_json(root / "preparation.json", {"completed_at": timestamp(), "limit": limit, "fingerprint_population": "train only", "training_cases": len(train), "development_cases": len(dev), "held_out_test_used": False, "source_commit": "858c26bc745d094f00c3eb03bdf795fba6588ac9"})


def train(run_id, epochs, steps, resume):
    root = paths(run_id)
    assert not (root / "training_status.json").exists(), "Completed training requires a new run ID"
    request = {"epochs_override": epochs, "steps_override": steps, "seed": 2025, "checkpoint_interval_epochs": 1}
    request_path = root / "training_request.json"
    if request_path.exists():
        assert resume and read_json(request_path) == request, "Use resume with the original configuration"
    write_json(request_path, request)
    from nnunetv2.training.nnUNetTrainer.nnUNetCLSTrainer import nnUNetCLSTrainerMTL
    directory = root / "preprocessed/Dataset905_LUNA25"
    torch.manual_seed(2025)
    np.random.seed(2025)
    torch.set_num_threads(4)
    trainer = nnUNetCLSTrainerMTL(read_json(directory / "nnUNetPlans.json"), "3d_fullres", 0, read_json(directory / "dataset.json"), device=torch.device("cuda"))
    if epochs is not None:
        trainer.num_epochs = epochs
    if steps is not None:
        trainer.num_iterations_per_epoch = steps
        trainer.configuration_manager.configuration["batch_size"] = 2
    trainer.initialize()
    trainer.save_every = 1
    if resume:
        trainer.load_checkpoint(str(Path(trainer.output_folder) / "checkpoint_latest.pth"))
    trainer.run_training()
    checkpoint = Path(trainer.output_folder) / "checkpoint_final.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer.network.load_state_dict(state["network_weights"], strict=True)
    write_json(root / "training_status.json", {"status": "completed", "completed_at": timestamp(), "checkpoint": str(checkpoint), "epochs": trainer.num_epochs, "iterations_per_epoch": trainer.num_iterations_per_epoch, "checkpoint_reload": "passed", "purpose": "smoke" if steps is not None else "training"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "train"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.run_id, args.limit)
    else:
        train(args.run_id, args.epochs, args.steps, args.resume)

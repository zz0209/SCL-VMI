import argparse
import importlib.util
import time
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.frozen_features import SUITE
from sclvmi.head import feature_table, metrics


def fmcib_upstream():
    storage, _, _ = context()
    root = Path(storage["runs"])
    output = root / SUITE / "fmcib_upstream_linear"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        print(read_json(output / "result.json"), flush=True)
        return
    original = read_json(root / "20260925_nodule_baselines_s2025_fmcib/result.json")
    directory = Path(original["feature_directory"])
    train, train_x = feature_table(directory, "train")
    dev, dev_x = feature_table(directory, "development")
    assert set(train.PatientID).isdisjoint(dev.PatientID)
    source_path = ROOT / "third_party/fmcib_source/experiments/adaptation/linear/common.py"
    spec = importlib.util.spec_from_file_location("fmcib_upstream_linear", source_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    scaler = StandardScaler().fit(train_x)
    x, v = scaler.transform(train_x), scaler.transform(dev_x)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2025), storage=f"sqlite:///{(output / 'search.db').as_posix()}", study_name="official_fmcib_linear", load_if_exists=True)
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        study.optimize(partial(upstream.objective, train_X=x, train_y=train.label.to_numpy(), val_X=v, val_y=dev.label.to_numpy(), scoring="roc_auc"), n_trials=max(0, 12-len(study.trials)), n_jobs=1, show_progress_bar=True)
        classifier = LogisticRegression(**study.best_params, random_state=upstream.SEED, max_iter=1000).fit(x, train.label)
    head = make_pipeline(scaler, classifier)
    joblib.dump(head, output / "head.joblib")
    p = head.predict_proba(dev_x)[:, 1]
    np.testing.assert_array_equal(p, joblib.load(output / "head.joblib").predict_proba(dev_x)[:, 1])
    dev.assign(probability=p).to_csv(output / "development_predictions.csv", index=False)
    study.trials_dataframe().to_csv(output / "trials.csv", index=False)
    result = {"status": "completed", "model": "fmcib", "arm": "upstream_linear", "run_id": output.name, "source": str(source_path), "source_sha256": sha256(source_path), "feature_directory": str(directory), "metrics": metrics(dev.label, p), "best_params": study.best_params, "seconds": time.perf_counter()-started, "trials": 12, "seed": 2025, "completed_at": timestamp(), "test_used": False, "adaptations": ["fixed project train/development split", "seeded Optuna sampler", "12 trials", "independent test remains unused"]}
    write_json(output / "result.json", result)
    print(result, flush=True)


def eva_smoke():
    from eva import core
    from eva.vision.models.networks.abmil import ABMIL
    from sclvmi.frozen_heads import load_features

    storage, _, _ = context()
    root = Path(storage["runs"]) / SUITE
    directory = Path(read_json(root / "coralbay_smoke.json")["directory"])
    frames, bags, _ = load_features(directory, limit=16)
    train_x = torch.from_numpy(bags[0])
    mean = train_x.mean((0, 1), keepdim=True)
    scale = train_x.std((0, 1), keepdim=True).clamp_min(1e-4)
    train_x = (train_x-mean)/scale
    labels = torch.tensor(frames[0].label.to_numpy(), dtype=torch.float32)
    output = root / "eva_framework_smoke"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2025)
    head = ABMIL(input_size=train_x.shape[-1], output_size=1, projected_input_size=128, hidden_sizes_mlp=(64,), pad_value=None)
    initial = [p.detach().clone() for p in head.parameters()]
    module = core.HeadModule(head=head, criterion=nn.BCEWithLogitsLoss(), optimizer=partial(torch.optim.AdamW, lr=3e-4))
    loader = DataLoader(TensorDataset(train_x, labels), batch_size=4, shuffle=True, num_workers=0)
    trainer = core.Trainer(accelerator="cpu", max_steps=8, default_root_dir=str(output), logger=False, enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(module, train_dataloaders=loader)
    assert trainer.global_step == 8
    assert any(not torch.equal(before, after) for before, after in zip(initial, head.parameters()))
    assert all(torch.isfinite(p).all() for p in head.parameters())
    trainer.save_checkpoint(output / "eva_head.ckpt")
    saved = torch.load(output / "eva_head.ckpt", weights_only=False)
    np.testing.assert_array_equal(saved["state_dict"]["head.projector.0.weight"].numpy(), head.projector[0].weight.detach().numpy())
    write_json(output / "validation.json", {"status": "passed", "data": "real CoralBay training-case features", "cases": len(labels), "steps": trainer.global_step, "weights_changed": True, "finite_weights": True, "checkpoint_exact": True, "framework": "kaiko-eva", "commit": "e43e74a99b75660b0014f790f25a33dd9f11e121", "completed_at": timestamp()})


def fmcib_inference():
    from sclvmi.predict import fm_predict

    storage, _, _ = context()
    output = Path(storage["runs"]) / SUITE / "fmcib_upstream_linear"
    cases = pd.read_csv(output / "development_predictions.csv").groupby("label", group_keys=False).head(2)
    torch.set_num_threads(4)
    probabilities = [fm_predict("fmcib", output, row.image)["malignancy_probability"] for row in cases.itertuples()]
    np.testing.assert_allclose(probabilities, cases.probability, rtol=1e-4, atol=1e-5)
    receipt = {"status":"passed", "cases":len(cases), "max_absolute_error":float(np.max(np.abs(probabilities-cases.probability.to_numpy()))), "completed_at":timestamp()}
    write_json(output / "image_inference_verification.json", receipt)
    print("FMCIB upstream raw-image inference", receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["fmcib", "eva", "fmcib-inference"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        {"fmcib":fmcib_upstream, "eva":eva_smoke, "fmcib-inference":fmcib_inference}[args.task]()

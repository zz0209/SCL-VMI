import hashlib
import json
import os
import socket
import time
from pathlib import Path

from .context import ROOT, context, read_json, timestamp, write_json


def configuration():
    return read_json(ROOT / "configs/sae_campaign.json")


def campaign_root(value=None):
    if value is not None:
        return Path(value)
    storage, _, _ = context()
    return Path(storage["runs"]) / configuration()["campaign"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pause_requested(root, worker):
    path = Path(root) / "controls.json"
    if not path.exists():
        return False
    control = read_json(path, synchronized=True)
    return bool(control.get("all_paused") or control.get("workers", {}).get(worker, {}).get("paused"))


def progress(root, task, state, completed, total, started, **extra):
    value = {
        "task": task, "state": state, "completed": completed, "total": total,
        "elapsed_seconds": time.monotonic() - started, "updated_at": timestamp(),
        "host": socket.gethostname(), "pid": os.getpid(), **extra,
    }
    write_json(Path(root) / "progress" / f"{task}.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def ensure_identity(path, value):
    path = Path(path)
    if path.exists():
        assert read_json(path) == value, f"Changed identity requires a new asset: {path}"
    else:
        write_json(path, value)


def training_identity(config, job, matrix):
    return {
        "campaign": config["campaign"], "run_id": job["run_id"],
        "model": job["model"], "site": job["site"], "seed": job["seed"],
        "input_dim": matrix["channels"], "dict_size": matrix["channels"] * job["expansion"],
        "expansion": job["expansion"], "k": job["k"],
        "training": {**config["training"], **job.get("training", {})},
        "matrix_identity": matrix["identity"], "site_specification": config["models"][job["model"]][job["site"]],
        "inference": {**config["inference"], **job.get("inference", {})}, "acceptance": config["acceptance"],
        "upstream": config["upstream"], "test_used": False,
    }

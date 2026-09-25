import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with FileLock(str(path) + ".lock", timeout=10):
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def context():
    storage = read_json(os.environ.get("SCLVMI_STORAGE", ROOT / "configs/storage.local.json"))
    config = read_json(ROOT / "configs/pipeline.json")
    os.environ["HF_HOME"] = storage["hf_home"]
    os.environ["HF_HUB_CACHE"] = storage["hf_hub_cache"]
    os.environ["HF_XET_CACHE"] = storage["hf_xet_cache"]
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["nnUNet_compile"] = "false"
    dataset = Path(storage["dataset_root"]) / "snapshots" / config["dataset_revision"] / config["dataset"]
    return storage, config, dataset


def timestamp():
    return datetime.now(timezone.utc).isoformat()

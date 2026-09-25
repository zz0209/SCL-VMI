import hashlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .data import load_manifest, select_cases
from .models import load_encoder, spatial_features
from .preprocessing import preprocess


def preprocessed_cases(rows, name):
    if name == "fmcib":
        for row in rows:
            yield row, preprocess(row.image, name)
        return
    with ThreadPoolExecutor(max_workers=2) as executor:
        iterator = iter(rows)
        pending = deque((row, executor.submit(preprocess, row.image, name)) for row in islice(iterator, 2))
        while pending:
            row, future = pending.popleft()
            image = future.result()
            for following in islice(iterator, 1):
                pending.append((following, executor.submit(preprocess, following.image, name)))
            yield row, image


def extract(name, splits, limit=None, spatial=False):
    storage, config, _ = context()
    model, assets = load_encoder(name)
    asset_identity = {"revision": assets.get("revision", "10528450"), "weights": [{"path": item["path"], "sha256": item["sha256"]} for item in assets["downloaded"] if item["path"].endswith((".pt", ".safetensors", ".bin"))]} if "downloaded" in assets else {"url": assets["url"], "sha256": assets["sha256"]}
    signature = {"model": name, "assets": asset_identity, "preprocessing": config["models"][name], "revision": config["dataset_revision"], "code": {file: sha256(ROOT / "src/sclvmi" / file) for file in ["models.py", "preprocessing.py"]}, "precision": "float32", "spatial": spatial}
    identity = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    directory = Path(storage["activations"]) / "luna25" / name / identity
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", signature)
    model = model.cuda()
    table = load_manifest()
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    completed = 0
    for split in splits:
        cases = select_cases(table, split, limit)
        pending_rows = []
        for row in cases.itertuples():
            path = directory / f"{row.identifier}.npz"
            if path.exists():
                with np.load(path) as cached:
                    assert np.isfinite(cached["embedding"]).all()
                continue
            pending_rows.append(row)
        print(f"{name} {split}: reusing {len(cases) - len(pending_rows)}/{len(cases)} cached cases", flush=True)
        for row, image in tqdm(preprocessed_cases(pending_rows, name), total=len(pending_rows), desc=f"{name} {split}", mininterval=2):
            path = directory / f"{row.identifier}.npz"
            image = image.unsqueeze(0).cuda()
            with torch.inference_mode():
                features = spatial_features(model, name, image)
                embedding = features.mean(dim=(2, 3, 4)).float().cpu().numpy()[0]
            assert np.isfinite(embedding).all() and np.linalg.norm(embedding) > 0
            payload = {"embedding": embedding, "spatial_shape": np.array(features.shape), "input_shape": np.array(image.shape)}
            if spatial:
                payload["spatial"] = features.float().cpu().numpy()[0]
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            temporary.rename(path)
            completed += 1
        cases.to_csv(directory / f"{split}_members.csv", index=False)
    receipt = {"completed_at": timestamp(), "new_cases": completed, "seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "splits": splits, "limit": limit, "directory": str(directory)}
    write_json(directory / f"receipt_{time.time_ns()}.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return directory

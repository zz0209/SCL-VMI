import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import nibabel as nib
import numpy as np
import torch

from sclvmi.context import context, timestamp, write_json
from sclvmi.data import build_manifest, load_manifest


def main():
    storage, _, _ = context()
    build_manifest()
    row = load_manifest().query("split == 'train'").iloc[0]
    volume = nib.load(row.image).get_fdata(dtype=np.float32)
    tensor = torch.tensor(volume[:16, :16, :16]).unsqueeze(0).unsqueeze(0).cuda() / 2048
    torch.manual_seed(2025)
    model = torch.nn.Conv3d(1, 4, 3).cuda()
    values = model(tensor)
    loss = values.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(values).all() and torch.isfinite(model.weight.grad).all()
    assert model.weight.grad.abs().sum() > 0
    spec = json.loads((ROOT / "configs/environments/pipeline.json").read_text())
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]
    result = {"completed_at": timestamp(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda, "shape": list(values.shape), "loss": float(loss.detach()), "source": "real training CT crop", "environment_spec_hash": spec_hash}
    write_json(Path(storage["logs"]) / "pipeline_environment_witness.json", result)
    print("WITNESS", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

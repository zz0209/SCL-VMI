import numpy as np
import torch

from sclvmi.context import read_json, timestamp, write_json
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_campaign import campaign_root
from sclvmi.sae_campaign_train import fit_job


root = campaign_root() / "smoke"
base = {"model": "vista", "site": "global", "seed": 2025, "k": 64, "expansion": 4, "matrix_directory": "smoke_matrices", "training": {"steps": 48, "batch_size": 64, "validate_every": 24, "checkpoint_every": 24, "sae_type": "topk", "checkpoint_selection": "qualified_fvu"}, "inference": {"calibration_train_tokens": 256}}
complete = {**base, "run_id": "smoke_vista_global_topk_complete_s2025"}
resumed = {**base, "run_id": "smoke_vista_global_topk_resume_s2025"}
reference = fit_job(root, complete, "smoke")
if not (root / "runs" / resumed["run_id"] / "result.json").exists():
    paused = fit_job(root, resumed, "smoke", stop_after=23)
    assert paused["state"] == "paused"
recovered = fit_job(root, resumed, "smoke")
first = torch.load(root / "runs" / complete["run_id"] / "resume.pt", map_location="cpu", weights_only=False)
second = torch.load(root / "runs" / resumed["run_id"] / "resume.pt", map_location="cpu", weights_only=False)
for key in first["state_dict"]:
    torch.testing.assert_close(first["state_dict"][key], second["state_dict"][key], rtol=0, atol=0)
assert reference["development"] == recovered["development"]
dictionary = SpatialDictionary(root / "runs" / complete["run_id"] / "dictionary.pt")
raw = torch.from_numpy(np.load(root / "smoke_matrices/vista/global/development.npy")).cuda()
with torch.inference_mode():
    pre = torch.relu(((raw - dictionary.mean) / dictionary.scale - dictionary.weights["b_dec"]) @ dictionary.weights["W_enc"])
    indices = torch.argsort(pre, dim=-1, descending=True)[:, :64]
    expected = torch.zeros_like(pre).scatter(-1, indices, torch.gather(pre, -1, indices))
    actual = dictionary.encode(raw)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
assert int((actual > 0).sum(1).max()) <= 64
receipt = {"state": "passed", "input": "actual VISTA global training and development vectors", "exact_checkpoint_recovery": True, "sorted_activation_agreement": True, "maximum_active_features": int((actual > 0).sum(1).max()), "created_at": timestamp()}
write_json(root / "topk_witness.json", receipt)
print("TOPK_WITNESS", receipt, flush=True)

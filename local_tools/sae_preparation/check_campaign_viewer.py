import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from sclvmi.context import ROOT, read_json, timestamp, write_json
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_campaign import campaign_root


sys.path.insert(0, str(ROOT / "local_tools/viewers/feature_explorer"))
from campaign_api import compute_response, native_input, volume_data


@torch.inference_mode()
def encoding_bounds(sae, raw, encoded):
    tokens = torch.from_numpy(raw).flatten(1).T
    pre = torch.relu(((tokens - sae.mean) / sae.scale - sae.weights["b_dec"]) @ sae.weights["W_enc"])
    tolerance = 2e-5
    if sae.request["training"].get("sae_type") == "topk":
        k = sae.request["k"]
        values = torch.topk(pre, k + 1, dim=1).values
        close = values[:, k - 1] - values[:, k] <= 2 * tolerance
        ambiguous = close[:, None] & ((pre - values[:, k - 1, None]).abs() <= 2 * tolerance)
    else:
        ambiguous = (pre - sae.threshold).abs() <= tolerance
    uncertain = ambiguous.T.numpy().reshape(encoded.shape)
    upper = np.where(uncertain, pre.T.numpy().reshape(encoded.shape), encoded)
    lower = np.where(uncertain, 0, encoded)
    return lower, upper, uncertain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--responses", action="store_true")
    args = parser.parse_args()
    root = campaign_root()
    records = []
    for run_id in args.runs:
        run = root / "runs" / run_id
        request = read_json(run / "request.json")
        sae = SpatialDictionary(run / "dictionary.pt", device="cpu")
        for case in [0, 127]:
            raw = native_input(request["model"], request["site"], case)
            volume, affine, canonical = volume_data(request["model"], case)
            with torch.inference_mode():
                encoded = sae.spatial_activations(torch.from_numpy(raw)).numpy()
            lower, upper, uncertain = encoding_bounds(sae, raw, encoded)
            changed_decision_features = []
            if args.responses:
                with h5py.File(root / "viewer/responses" / run_id / "responses.h5", "r") as saved:
                    assert saved["completed"][...].all()
                    flat = encoded.reshape(len(encoded), -1)
                    low, high = lower.reshape(flat.shape), upper.reshape(flat.shape)
                    for key, low_values, high_values in [("maximum", low.max(1), high.max(1)), ("mean", low.mean(1), high.mean(1))]:
                        values = saved[key][case]
                        assert np.all(values >= low_values - 3e-5) and np.all(values <= high_values + 3e-5), key
                    counts = np.rint(saved["active_fraction"][case] * flat.shape[1])
                    assert np.all(counts >= (low > 0).sum(1)) and np.all(counts <= (high > 0).sum(1))
                    changed_decision_features = np.flatnonzero(counts != (flat > 0).sum(1)).tolist()
            features = {0, request["dict_size"] - 1, int(encoded.reshape(len(encoded), -1).max(1).argmax())}
            features.update(changed_decision_features)
            ambiguous_features = np.flatnonzero(uncertain.reshape(len(encoded), -1).any(1))
            if len(ambiguous_features):
                features.add(int(ambiguous_features[0]))
            features = sorted(features)
            for feature in features:
                actual = compute_response(raw, sae, feature, request["site_specification"], volume, affine, canonical)
                stable = ~uncertain[feature]
                np.testing.assert_allclose(actual["native"][stable], encoded[feature][stable], rtol=2e-5, atol=2e-5)
                assert np.all(actual["native"] >= lower[feature] - 3e-5) and np.all(actual["native"] <= upper[feature] + 3e-5)
                if request["site"] == "global":
                    assert actual["dense"] is None and actual["native"].size == 1
                else:
                    voxels = np.indices(volume.shape, dtype=np.float64).reshape(3, -1).T
                    physical = nib.affines.apply_affine(canonical, voxels)
                    encoder_voxels = nib.affines.apply_affine(np.linalg.inv(affine), physical)
                    coordinates = encoder_voxels / request["site_specification"]["stride"]
                    shape = np.asarray(encoded[feature].shape)
                    normalized = coordinates / (shape - 1) * 2 - 1
                    grid = torch.tensor(normalized[:, ::-1].copy(), dtype=torch.float32).reshape(1, *volume.shape, 3)
                    expected = F.grid_sample(torch.from_numpy(actual["native"])[None, None], grid, align_corners=True, padding_mode="border", mode="bilinear")[0, 0].numpy()
                    np.testing.assert_allclose(actual["dense"], expected, rtol=3e-5, atol=3e-5)
                    assert np.isfinite(actual["dense"]).all()
                records.append({"run_id": run_id, "case": case, "feature": feature, "native_shape": list(actual["native"].shape), "maximum": float(actual["native"].max()), "threshold_near_positions": int(uncertain[feature].sum()), "total_positions": int(uncertain[feature].size)})
            print("VIEWER_WITNESS", run_id, case, features, flush=True)
    receipt = {"state": "passed", "response_statistics_checked": args.responses, "threshold_precision": "Validate stable decisions directly; bound discontinuous threshold decisions using pre-activations within 2e-5 and save their counts. TopK ties use twice this bound.", "checks": ["single-feature CPU encoding matches full dictionary encoding outside measured threshold roundoff bounds", "saved response statistics lie within the same bounds", "physical-coordinate interpolation matches independent torch grid_sample", "global dictionaries return one crop-level response"], "records": records, "created_at": timestamp()}
    write_json(root / "viewer" / f"validation_{time.time_ns()}.json", receipt)
    print("VIEWER_WITNESS_PASSED", len(records), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(3)
    main()

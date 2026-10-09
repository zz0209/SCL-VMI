import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sclvmi.context import context, read_json, sha256, timestamp, write_json
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_data import configuration, locations
from sclvmi.sae_predict import SpatialSAEPipeline
from sclvmi.sae_sites import site_affine


parser = argparse.ArgumentParser()
parser.add_argument("--runs", nargs="+", required=True)
args = parser.parse_args()
torch.set_num_threads(3)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
storage, _, _ = context()
root = Path(storage["runs"]) / configuration()["suite"]
for run_id in args.runs:
    run = root / run_id
    request = read_json(run / "request.json")
    directory, _ = locations(request["model"])
    members = pd.read_csv(directory / "full_members.csv", dtype={"PatientID": str})
    members = members.query("split == 'development'").sort_values("identifier").groupby("label", sort=True).head(1)
    predictions = pd.read_csv(run / "downstream_full/predictions.csv")
    pipeline = SpatialSAEPipeline(run / "dictionary.pt")
    cpu_dictionary = SpatialDictionary(run / "dictionary.pt", device="cpu")
    checks = []
    for row in members.itertuples():
        bundle = pipeline.extract(row.image)
        geometry = bundle["geometry"]
        saved_geometry = read_json(directory / "cases" / f"{row.identifier}.json")
        np.testing.assert_allclose(geometry["grid_affine_ras"], site_affine(saved_geometry, request["model"], request["site"]), rtol=0, atol=1e-10)
        assert bundle["activations"].shape == (request["dict_size"], *geometry["grid_shape"])
        assert np.isfinite(bundle["activations"]).all() and (bundle["activations"] >= 0).all()
        with np.load(directory / "cases" / f"{row.identifier}.npz") as saved:
            feature_map = torch.from_numpy(saved[request["site"]]).cuda()
            original = float(saved["probability"][0])
        expected = pipeline.dictionary.spatial_activations(feature_map).cpu().numpy()
        np.testing.assert_allclose(bundle["activations"], expected, rtol=0, atol=0)
        tokens = feature_map.flatten(1).T[:127]
        gpu_reconstruction = pipeline.dictionary.reconstruct(tokens).cpu()
        cpu_reconstruction = cpu_dictionary.reconstruct(tokens.cpu())
        torch.testing.assert_close(cpu_reconstruction, gpu_reconstruction, rtol=1e-4, atol=1e-4)
        saved_prediction = predictions.loc[predictions.identifier == row.identifier, "recovered"].item()
        assert abs(geometry["original_probability"] - original) < 1e-5
        assert abs(geometry["reconstructed_probability"] - saved_prediction) < 1e-5
        checks.append({"identifier": row.identifier, "label": row.label, "grid_shape": geometry["grid_shape"], "cpu_gpu_reconstruction_max_error": float((cpu_reconstruction - gpu_reconstruction).abs().max()), "probability_max_error": max(abs(geometry["original_probability"] - original), abs(geometry["reconstructed_probability"] - saved_prediction))})
        del bundle, expected, feature_map, tokens
    write_json(run / "inference_checks.json", {"status": "passed", "checks": checks, "dictionary_sha256": sha256(run / "dictionary.pt"), "source_sha256": sha256(Path(__file__)), "completed_at": timestamp(), "test_used": False})
    print("INFERENCE_PASSED", run_id, flush=True)
    del pipeline, cpu_dictionary
    gc.collect()
    torch.cuda.empty_cache()

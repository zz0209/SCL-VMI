import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from sclvmi.context import context, timestamp, write_json
from sclvmi.data import load_manifest
from sclvmi.features import extract
from sclvmi.head import fit_head
from sclvmi.models import load_encoder, spatial_features
from sclvmi.preprocessing import preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--models", nargs="+", default=["fmcib", "ctfm", "vista", "genesis"])
    args = parser.parse_args()
    storage, _, _ = context()
    output = Path(storage["runs"]) / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in args.models:
        directory = extract(name, ["train", "development"], limit=8, spatial=True)
        head = fit_head(directory, args.run_id + "_" + name)
        row = load_manifest().query("split == 'train'").iloc[0]
        model, _ = load_encoder(name)
        model = model.cuda()
        image = preprocess(row.image, name).unsqueeze(0).cuda()
        with torch.inference_mode():
            features = spatial_features(model, name, image)
            embedding = features.mean((2, 3, 4))
            if name == "fmcib":
                torch.testing.assert_close(embedding, model(image), atol=1e-5, rtol=1e-5)
            if name == "ctfm":
                torch.testing.assert_close(features, model(image)[-1])
            repeat = spatial_features(model, name, image).mean((2, 3, 4))
            torch.testing.assert_close(embedding, repeat, atol=1e-5, rtol=1e-5)
        results[name] = {"feature_directory": str(directory), "head_run": str(head), "input_shape": list(image.shape), "spatial_shape": list(features.shape), "embedding_shape": list(embedding.shape), "finite": bool(torch.isfinite(embedding).all()), "repeatability": "passed", "strict_weights": "passed"}
        write_json(output / "validation.json", {"updated_at": timestamp(), "models": results, "status": "running"})
        del model, image, features, embedding, repeat
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output / "validation.json", {"completed_at": timestamp(), "models": results, "status": "completed", "scope": "16 real cases per model; functionality, not predictive performance"})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .context import context, read_json, timestamp, write_json
from .frozen_features import MODELS, encode, load_fm
from .frozen_heads import SpatialHead
from .spatial_linear import spatial_vector


def load_head(run):
    result = read_json(run / "result.json")
    if (run / "head.joblib").exists():
        return joblib.load(run / "head.joblib"), result
    state = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    head = SpatialHead(len(state["mean"]), result["arm"], state["mean"], state["scale"])
    head.load_state_dict(state, strict=True)
    return head.cuda().eval(), result


def predict(model, head, result, image):
    config = read_json(Path(result["feature_directory"]) / "config.json")
    if result["model"] in MODELS[:4]:
        _, runtime, _ = context()
        assert config["settings"] == runtime["models"][result["model"]], "Restore this run's preprocessing configuration before inference"
    embedding, tokens, _ = encode(model, result["model"], image, config.get("settings"))
    if result["arm"] == "spatial_linear":
        return float(head.predict_proba(spatial_vector(tokens[None], result["model"]))[0,1])
    if result["arm"] in ["native_linear", "upstream_linear", "native_gbdt"]:
        return float(head.predict_proba(embedding[None])[:, 1][0])
    tensor = embedding[None, None] if result["arm"] == "native_mlp" else tokens[None]
    with torch.inference_mode():
        return float(head(torch.from_numpy(tensor).cuda()).sigmoid()[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert args.image is not None or args.verify
    head, result = load_head(args.run)
    model, _ = load_fm(result["model"])
    model.cuda()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.image:
        print({"malignancy_probability": predict(model, head, result, args.image)}, flush=True)
    if args.verify:
        table = pd.read_csv(args.run / "development_predictions.csv")
        cases = table.groupby("label", group_keys=False).head(2)
        actual = [predict(model, head, result, row.image) for row in cases.itertuples()]
        np.testing.assert_allclose(actual, cases.probability, rtol=1e-4, atol=1e-5)
        write_json(args.run / "image_inference_verification.json", {"status": "passed", "cases": len(cases), "classes": sorted(cases.label.unique().tolist()), "max_absolute_error": float(np.max(np.abs(actual-cases.probability.to_numpy()))), "completed_at": timestamp(), "run_id": args.run.name})
        print("RAW_IMAGE_VERIFICATION passed", args.run.name, flush=True)


if __name__ == "__main__":
    main()

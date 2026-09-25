import argparse
import json
from pathlib import Path

import torch

from .adaptation import Classifier, last_block, setup
from .context import ROOT, context, read_json, sha256
from .models import load_encoder, spatial_features
from .preprocessing import preprocess


def predict(run_directory, image_path):
    run = Path(run_directory)
    result = read_json(run / "result.json")
    request = read_json(run / "request.json")
    for file, digest in request["code"].items():
        assert sha256(ROOT / "src/sclvmi" / file) == digest, f"Source changed: {file}"
    name, recipe = result["model"], result["recipe"]
    setup(recipe["seed"])
    if recipe["mode"] == "frozen":
        torch.backends.cudnn.allow_tf32 = True
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    head = Classifier(checkpoint["head"]["mean"], checkpoint["head"]["scale"], recipe["head"], recipe.get("hidden", 256), recipe.get("dropout", .1))
    head.load_state_dict(checkpoint["head"], strict=True)
    model, _ = load_encoder(name)
    if checkpoint["block"] is not None:
        last_block(model, name).load_state_dict(checkpoint["block"], strict=True)
    model, head = model.cuda().eval(), head.cuda().eval()
    image = preprocess(image_path, name).unsqueeze(0).cuda()
    with torch.inference_mode():
        features = spatial_features(model, name, image).mean((2, 3, 4))
        probability = float(head(features).sigmoid().item())
    return {"model": name, "arm": recipe["arm"], "malignancy_probability": probability, "checkpoint": str(run / "best.pt"), "input_shape": list(image.shape), "encoder_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "purpose": "research inference"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    storage, _, _ = context()
    print(json.dumps(predict(Path(storage["runs"])/args.run_id, args.image), indent=2))

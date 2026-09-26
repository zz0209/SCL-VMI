import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .context import read_json, timestamp, write_json
from .data import load_manifest, select_cases
from .frozen_features import encode, load_fm
from .frozen_predict import load_head
from .material_features import descriptor, load_case, prepare_tensor, stage_features
from .spatial_linear import spatial_vector


class MaterialPipeline:
    def __init__(self, run, kind):
        self.run = Path(run)
        self.kind = kind
        self.head, self.result = load_head(self.run)
        self.name = self.result["model"]
        self.config = read_json(Path(self.result["feature_directory"]) / "config.json")
        self.model = None
        self.material = read_json(self.run / "material_config.json") if kind == "descriptor" else None

    def extract(self, image):
        if self.model is None:
            self.model, _ = load_fm(self.name)
            self.model.cuda()
        bundle = {}
        if self.kind == "reference" or (self.kind == "descriptor" and self.material["descriptor"] in ["reference_meanmax", "two_view"]):
            cfg = self.config if self.kind == "reference" else read_json(Path(self.material["reference_directory"])/"config.json")
            embedding, tokens, _ = encode(self.model, self.name, image, cfg["settings"])
            bundle["reference"] = {"embedding": embedding, "tokens": tokens}
        if self.kind == "local_max" or (self.kind == "descriptor" and self.material["descriptor"] != "reference_meanmax"):
            tensor = prepare_tensor(image, self.name, self.config["settings"])
            bundle["local"] = stage_features(self.model, self.name, tensor)
        return bundle

    def cached_features(self, identifier):
        if self.kind == "reference":
            return {"reference": load_case(Path(self.result["feature_directory"]), identifier)}
        if self.kind == "local_max":
            return {"local": load_case(Path(self.result["feature_directory"]), identifier)}
        return {"reference": load_case(Path(self.material["reference_directory"]), identifier), "local": load_case(Path(self.material["local_directory"]), identifier)}

    def predict_features(self, bundle):
        if self.kind == "descriptor":
            x = descriptor(bundle.get("reference"), bundle.get("local"), self.material["descriptor"])
            return float(self.head.predict_proba(x[None])[0, 1])
        values = bundle["reference"] if self.kind == "reference" else bundle["local"]
        arm = self.result["arm"]
        if arm == "spatial_linear":
            return float(self.head.predict_proba(spatial_vector(values["tokens"][None], self.name))[0, 1])
        if arm in ["native_linear", "upstream_linear", "native_gbdt"]:
            return float(self.head.predict_proba(values["embedding"][None])[0, 1])
        x = values["embedding"][None, None] if arm == "native_mlp" else values["tokens"][None]
        with torch.inference_mode():
            return float(self.head(torch.from_numpy(x).cuda()).sigmoid()[0])

    def predict(self, image):
        return self.predict_features(self.extract(image))

    def training_features(self):
        frame = select_cases(load_manifest(), "train")
        for row in frame.itertuples():
            yield row.identifier, self.cached_features(row.identifier)

    def verify(self):
        frame = pd.read_csv(self.run / "development_predictions.csv", dtype={"PatientID": str})
        cases = frame.groupby("label", group_keys=False).head(2)
        actual, cached, seconds = [], [], []
        for row in cases.itertuples():
            started = time.perf_counter()
            actual.append(self.predict(row.image))
            seconds.append(time.perf_counter() - started)
            cached.append(self.predict_features(self.cached_features(row.identifier)))
        np.testing.assert_allclose(actual, cases.probability, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cached, cases.probability, rtol=1e-4, atol=1e-5)
        receipt = {"status": "passed", "kind": self.kind, "cases": len(cases), "classes": sorted(cases.label.unique().tolist()), "raw_max_error": float(np.max(np.abs(actual-cases.probability.to_numpy()))), "feature_max_error": float(np.max(np.abs(cached-cases.probability.to_numpy()))), "raw_seconds": seconds, "warm_median_seconds": float(np.median(seconds[1:])), "timing_scope": "First case includes model load; remaining three include image reading, preprocessing, encoder and classifier", "completed_at": timestamp()}
        write_json(self.run / "material_verification.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--kind", choices=["reference", "descriptor", "local_max"], required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = MaterialPipeline(args.run, args.kind)
    if args.verify:
        print(pipeline.verify(), flush=True)
    else:
        assert args.image is not None
        print({"malignancy_probability": pipeline.predict(args.image)}, flush=True)


if __name__ == "__main__":
    main()

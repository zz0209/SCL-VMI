import argparse
from pathlib import Path

import numpy as np
import torch

from .context import write_json
from .models import spatial_features
from .sae import SpatialDictionary
from .sae_sites import SITES, SpatialEncoder, volume_with_geometry


class SpatialSAEPipeline:
    def __init__(self, dictionary_path, device="cuda", encoder=None):
        self.dictionary = SpatialDictionary(dictionary_path, device=device)
        self.name = self.dictionary.request["model"]
        self.site = self.dictionary.request["site"]
        self.specification = self.dictionary.request["site_specification"] if "site_specification" in self.dictionary.request else {**SITES[self.name][self.site], "kind": "spatial"}
        self.encoder = SpatialEncoder(self.name, device=device) if encoder is None else encoder
        assert self.encoder.name == self.name and self.encoder.device == device

    @torch.inference_mode()
    def extract(self, image_path):
        tensor, geometry = volume_with_geometry(image_path, self.name)
        pooled = self.specification["kind"] == "pooled"
        captures = {}
        if not pooled:
            handle = self.encoder.model.get_submodule(self.specification["module"]).register_forward_hook(lambda module, inputs, output: captures.update(value=output.detach().clone()))
        final = spatial_features(self.encoder.model, self.name, tensor[None].to(self.encoder.device))
        if not pooled:
            handle.remove()
        feature_map = final[0].mean((1, 2, 3), keepdim=True) if pooled else captures["value"][0]
        activities = self.dictionary.spatial_activations(feature_map)
        recovered = self.dictionary.decode(activities.flatten(1).T).T.reshape_as(feature_map)
        original_probability = float(self.encoder.probability(final)[0])
        continued = final - feature_map[None] + recovered[None] if pooled else self.encoder.continue_from(self.site, recovered[None])
        reconstructed_probability = float(self.encoder.probability(continued)[0])
        affine = None if pooled else (np.asarray(geometry["input_affine_ras"]) @ np.diag([self.specification["stride"]] * 3 + [1])).tolist()
        geometry.update({"model": self.name, "site": self.site, "kind": self.specification["kind"], "grid_affine_ras": affine, "grid_shape": list(feature_map.shape[1:]), "run_id": self.dictionary.request["run_id"], "dictionary_size": self.dictionary.request["dict_size"], "threshold": self.dictionary.threshold, "original_probability": original_probability, "reconstructed_probability": reconstructed_probability})
        return {"input": tensor.cpu().numpy(), "activations": activities.cpu().numpy(), "geometry": geometry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix(".json").exists()
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle = SpatialSAEPipeline(args.dictionary).extract(args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez(handle, input=bundle["input"], activations=bundle["activations"])
    write_json(args.output.with_suffix(".json"), bundle["geometry"])
    print({key: bundle["geometry"][key] for key in ["model", "site", "run_id", "grid_shape", "dictionary_size"]}, flush=True)


if __name__ == "__main__":
    main()

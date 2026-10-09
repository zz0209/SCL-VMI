import argparse
from pathlib import Path

import numpy as np
import torch

from .context import write_json
from .sae import SpatialDictionary
from .sae_sites import SpatialEncoder, site_affine, volume_with_geometry


class SpatialSAEPipeline:
    def __init__(self, dictionary_path, device="cuda"):
        self.dictionary = SpatialDictionary(dictionary_path, device=device)
        self.name = self.dictionary.request["model"]
        self.site = self.dictionary.request["site"]
        self.encoder = SpatialEncoder(self.name, device=device)

    @torch.inference_mode()
    def extract(self, image_path):
        tensor, geometry = volume_with_geometry(image_path, self.name)
        outputs = self.encoder.extract(tensor, sites=[self.site])
        feature_map = outputs[self.site][0]
        activities = self.dictionary.spatial_activations(feature_map)
        recovered = self.dictionary.decode(activities.flatten(1).T).T.reshape_as(feature_map)
        original_probability = float(self.encoder.probability(outputs["final"])[0])
        reconstructed_probability = float(self.encoder.probability(self.encoder.continue_from(self.site, recovered[None]))[0])
        geometry.update({"model": self.name, "site": self.site, "grid_affine_ras": site_affine(geometry, self.name, self.site).tolist(), "grid_shape": list(feature_map.shape[1:]), "run_id": self.dictionary.request["run_id"], "dictionary_size": self.dictionary.request["dict_size"], "threshold": self.dictionary.threshold, "original_probability": original_probability, "reconstructed_probability": reconstructed_probability})
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

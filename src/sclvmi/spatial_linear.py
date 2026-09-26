import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .context import context, read_json
from .frozen_features import MODELS, SUITE
from .frozen_heads import fit_linear, load_features


GRIDS = {"fmcib": (2,2,2), "ctfm": (2,4,4), "vista": (3,3,3), "genesis": (8,8,4), "coralbay": (2,2,2)}


def spatial_vector(tokens, name):
    if name == "tapct":
        return tokens.reshape(len(tokens), -1)
    grid = GRIDS[name]
    assert tokens.shape[1] == np.prod(grid)
    tensor = torch.from_numpy(tokens).transpose(1,2).reshape(len(tokens), tokens.shape[-1], *grid)
    return F.adaptive_avg_pool3d(tensor, (2,2,2)).flatten(1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--variant")
    args = parser.parse_args()
    storage, _, _ = context()
    root = Path(storage["runs"]) / (SUITE if args.variant is None else SUITE+"_"+args.variant)
    directory = args.features or Path(read_json(Path(storage["runs"])/SUITE/f"{args.model}_features.json")["directory"])
    torch.set_num_threads(4)
    frames, bags, _ = load_features(directory)
    vectors = [spatial_vector(bag,args.model) for bag in bags]
    result = fit_linear(args.model,directory,frames,vectors,root,arm="spatial_linear")
    print(result,flush=True)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from sclvmi.context import timestamp, write_json
from sclvmi.sae import SpatialDictionary
from interpretation_whole import CONFIG, SETTINGS, Encoders, audit, dictionaries, geometry, output_root, runs_root, window_affine, window_input
from whole import load_case, sample_input


def native_index(model, site, value, tile, position):
    spec = SETTINGS["sites"][f"{model}:{site}"]
    world = value["origin"] + position * value["spacing"]
    start = value["tile_origin"] + tile * value["tile_step"]
    voxel = np.linalg.solve(window_affine(model, start, spec["window_offset"]), np.r_[world, 1])[:3]
    native = voxel / spec["stride"]
    assert np.allclose(native, np.rint(native), atol=1e-6), native
    native = np.rint(native).astype(int)
    assert np.all(native >= spec["first"]) and np.all(native < spec["first"] + spec["count"]), native
    return start, native


@torch.inference_mode()
def check(model, case, samples, source):
    rows = audit()["cases"]
    volume, affine = load_case(rows[case])
    encoders = Encoders(model)
    results = []
    generator = np.random.default_rng(20261015 + case)
    fmcib_window = SETTINGS["sites"]["fmcib:layer2"]["window"]
    if model == "fmcib":
        start = affine[:3, 3] + np.asarray(volume.shape) * np.diag(affine)[:3] / 2
        np.testing.assert_array_equal(window_input(volume, affine, "fmcib", start, fmcib_window, 0), sample_input(volume, affine, "fmcib", start))
        cube = torch.from_numpy(window_input(volume, affine, "fmcib", start, 84, -16))[None].cuda()
        deployed = torch.from_numpy(window_input(volume, affine, "fmcib", start, fmcib_window, 0))[None].cuda()
        context = encoders.features("trained", cube)["layer1_context"]
        layer1 = encoders.trained.layer1(encoders.trained.maxpool(encoders.trained.act(encoders.trained.bn1(encoders.trained.conv1(deployed)))))
        correspondence = float((context[:, :, 9:13, 9:13, 9:13] - layer1[:, :, 5:9, 5:9, 5:9]).abs().max())
        assert correspondence < 1e-4, correspondence
        results.append({"check": "complete-context cube interior equals deployed-window interior", "max_abs_difference": correspondence})
    for site in (["layer1_context", "layer2"] if model == "fmcib" else ["stage2", "stage3"]):
        value = geometry(model, site, volume.shape, affine)
        spec = SETTINGS["sites"][f"{model}:{site}"]
        for encoder, run in dictionaries(model, site):
            path = output_root() / source / run / f"case_{case:02d}.h5"
            if not path.with_suffix(".json").exists():
                continue
            sae = SpatialDictionary(runs_root() / CONFIG["campaign"] / "runs" / run / "dictionary.pt")
            with h5py.File(path, "r") as saved:
                written = int(saved.attrs["completed_tiles"])
                tiles = np.asarray(saved.attrs["tiles"])
                indices = np.arange(int(np.prod(tiles)))
                if source == "smoke_maps":
                    center = np.ravel_multi_index(tuple(tiles // 2), tiles)
                    indices = indices[center:center + written]
                chosen = generator.choice(indices[:written], size=min(samples, written), replace=False)
                differences = []
                for index in chosen:
                    tile = np.asarray(np.unravel_index(index, tiles))
                    inner = generator.integers(0, value["core"], size=3)
                    position = np.minimum(tile * value["core"] + inner, value["shape"] - 1)
                    tile = position // value["core"]
                    start, native = native_index(model, site, value, tile, position)
                    batch = torch.from_numpy(window_input(volume, affine, model, start, spec["window"], spec["window_offset"]))[None].cuda()
                    activations = encoders.features(encoder, batch)[site][0, :, native[0], native[1], native[2]]
                    expected = sae.encode(activations[None])[0].cpu().numpy()
                    if "valid" in saved and not saved["valid"][tuple(position)]:
                        expected = np.zeros_like(expected)
                    observed = saved["values"][(slice(None), *position)].astype(np.float32)
                    differences.append(float(np.max(np.abs(observed - expected.astype(np.float16).astype(np.float32)))))
                maximum = max(differences)
                assert maximum < 1e-2, (run, maximum)
                results.append({"run_id": run, "case": case, "positions": len(differences), "max_abs_difference": maximum, "valid_fraction": float(saved["valid"][...].mean()) if "valid" in saved else 1.0})
                print(results[-1], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fmcib", "vista"], required=True)
    parser.add_argument("--cases", default="0")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--source", default="maps")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [row for case in args.cases.split(",") for row in check(args.model, int(case), args.samples, args.source)]
    stamp = timestamp().replace(":", "").replace("+", "_")
    write_json(output_root() / "verification" / f"witness_{args.model}_{args.source}_{stamp}.json", {"state": "passed", "rows": rows, "rule": "map value equals an independent window, encoder and dictionary computation within float16 rounding; complete-context cube interior equals the deployed-window interior", "created_at": timestamp(), "test_used": False})
    print({"state": "passed", "checks": len(rows)}, flush=True)


if __name__ == "__main__":
    main()

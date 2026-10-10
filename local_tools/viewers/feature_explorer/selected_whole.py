import argparse
from contextlib import ExitStack
from pathlib import Path
import time

import h5py
import nibabel as nib
import numpy as np
import torch

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.models import load_encoder, spatial_features
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_campaign import campaign_root, ensure_identity, pause_requested, progress
from whole import input_affine, load_case, sample_input


def configuration():
    return read_json(ROOT / "configs/sae_whole_viewer.json")


def output_root():
    return Path(context()[0]["runs"]) / configuration()["run_id"]


def selected_runs(model):
    selection = read_json(campaign_root() / "selection.json")
    assert selection["state"] == "completed"
    return [run for group in selection["selected"] if group["model"] == model for run in group["replicas"]]


def map_geometry(model, site, shape, affine):
    settings = configuration()["models"][model]
    step = settings["region_step_mm"]
    extent = (np.asarray(shape) - 1) * nib.affines.voxel_sizes(affine)
    tiles = np.floor(extent / step).astype(int) + 1
    origin = affine[:3, 3].copy()
    if site == "global":
        return {"tiles": tiles, "shape": tiles, "origin": origin, "spacing": float(step), "core": 1, "tile_origin": origin, "tile_step": step}
    first, stop = settings["core"][site]
    core = stop - first
    stride = 4 if core == 8 else 8
    shift = np.eye(4)
    shift[:3, 3] = first
    native_affine = input_affine(model, origin) @ np.diag([stride] * 3 + [1]) @ shift
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(native_affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
    canonical_affine = native_affine @ nib.orientations.inv_ornt_aff(transform, (core,) * 3)
    assert np.allclose(canonical_affine[:3, :3], np.eye(3) * (step / core))
    native_origin = canonical_affine[:3, 3]
    native_shape = np.floor((origin + extent - native_origin) / (step / core)).astype(int) + 1
    assert np.all(native_shape > 0) and np.all(native_shape <= tiles * core)
    return {"tiles": tiles, "shape": native_shape, "origin": native_origin, "spacing": step / core, "core": core, "tile_origin": origin, "tile_step": step}


class SelectedEncoder:
    def __init__(self, name):
        self.name = name
        self.model, self.assets = load_encoder(name)
        self.model.cuda().eval()
        self.runs = selected_runs(name)
        self.dictionaries = {run: SpatialDictionary(campaign_root() / "runs" / run / "dictionary.pt") for run in self.runs}
        self.captures = {}
        self.handles = []
        requests = {sae.request["site"]: sae.request for sae in self.dictionaries.values()}
        for site, request in requests.items():
            if site == "global":
                continue
            def capture(module, inputs, value, key=site):
                self.captures[key] = value.detach()
            self.handles.append(self.model.get_submodule(request["site_specification"]["module"]).register_forward_hook(capture))

    @torch.inference_mode()
    def extract(self, inputs):
        final = spatial_features(self.model, self.name, torch.from_numpy(np.stack(inputs)).cuda())
        self.captures["global"] = final.mean((2, 3, 4), keepdim=True)
        result = {}
        for site, values in self.captures.items():
            if site != "global":
                first, stop = configuration()["models"][self.name]["core"][site]
                values = values[:, :, first:stop, first:stop, first:stop]
            result[site] = values.contiguous()
        self.captures.clear()
        return result

    @torch.inference_mode()
    def encode(self, run, values):
        sae = self.dictionaries[run]
        batch, _, x, y, z = values.shape
        tokens = values.permute(0, 2, 3, 4, 1).reshape(-1, values.shape[1])
        encoded = sae.encode(tokens).reshape(batch, x, y, z, -1).permute(0, 4, 1, 2, 3)
        if self.name == "fmcib" and sae.request["site"] != "global":
            encoded = encoded.permute(0, 1, 4, 3, 2).flip((2, 3))
        return encoded.cpu().numpy()


def initialize(saved, run, geometry, count, case):
    if "values" in saved:
        assert saved.attrs["run_id"] == run and saved.attrs["total_tiles"] == count
        return
    request = read_json(campaign_root() / "runs" / run / "request.json")
    core = geometry["core"]
    saved.create_dataset("values", (request["dict_size"], *geometry["shape"]), dtype="f2", chunks=(min(32, request["dict_size"]), *np.minimum(core, geometry["shape"])), compression="lzf", shuffle=True)
    saved.create_dataset("maximum", data=np.zeros(request["dict_size"], np.float32))
    saved.attrs.update(run_id=run, completed_tiles=0, total_tiles=count, status="running", case_index=case,
                       shape=geometry["shape"], tiles=geometry["tiles"], origin=geometry["origin"], step_mm=geometry["spacing"],
                       core_size=core, tile_origin=geometry["tile_origin"], tile_step_mm=geometry["tile_step"],
                       dictionary_sha256=sha256(campaign_root() / "runs" / run / "dictionary.pt"))
    saved.flush()


def run_model(model, smoke=False):
    config = configuration()
    source = Path(context()[0]["runs"]) / config["source_run"] / "source_audit.json"
    audit = read_json(source)
    root = output_root() / ("smoke" if smoke else "maps")
    root.mkdir(parents=True, exist_ok=True)
    identity = {"configuration": config, "source_audit_sha256": sha256(source), "selection_sha256": sha256(campaign_root() / "selection.json"), "code_sha256": sha256(Path(__file__)), "input_code_sha256": sha256(Path(__file__).with_name("whole.py")), "smoke": smoke, "test_used": False}
    ensure_identity(root / "request.json", identity)
    encoder = SelectedEncoder(model)
    batch_size = config["batch_size"]
    for case, row in enumerate(audit["cases"][:1] if smoke else audit["cases"]):
        volume, affine = load_case(row)
        geometries = {run: map_geometry(model, sae.request["site"], volume.shape, affine) for run, sae in encoder.dictionaries.items()}
        reference = next(iter(geometries.values()))
        tiles = reference["tiles"]
        count = int(np.prod(tiles))
        indices = np.arange(count)
        if smoke:
            center = np.ravel_multi_index(tuple(tiles // 2), tiles)
            indices = np.unique(np.r_[indices[:2], indices[center:center + 2], indices[-2:]])
        started = time.monotonic()
        paths = {}
        with ExitStack() as stack:
            outputs = {}
            for run in encoder.runs:
                path = root / run / f"case_{case:02d}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                paths[run] = path
                saved = stack.enter_context(h5py.File(path, "a"))
                initialize(saved, run, geometries[run], len(indices), case)
                outputs[run] = saved
            completed = min(int(saved.attrs["completed_tiles"]) for saved in outputs.values())
            initial = completed
            for offset in range(completed, len(indices), batch_size):
                task = f"whole-selected-{model}"
                if pause_requested(campaign_root(), "local"):
                    progress(campaign_root(), task, "paused", case * count + offset, len(audit["cases"]) * count, started, worker="local")
                    write_json(output_root() / f"progress_{model}.json", {"state": "paused", "case": case, "completed": offset, "total": len(indices), "updated_at": timestamp()})
                    raise SystemExit(75)
                selected = indices[offset:offset + batch_size]
                coordinates = np.asarray(np.unravel_index(selected, tiles)).T
                starts = coordinates * reference["tile_step"] + affine[:3, 3]
                values = encoder.extract([sample_input(volume, affine, model, position) for position in starts])
                for run, saved in outputs.items():
                    stop = offset + len(selected)
                    if int(saved.attrs["completed_tiles"]) >= stop:
                        continue
                    assert int(saved.attrs["completed_tiles"]) == offset
                    geometry = geometries[run]
                    encoded = encoder.encode(run, values[encoder.dictionaries[run].request["site"]])
                    compressed = encoded.astype(np.float16)
                    assert np.isfinite(compressed).all() and (compressed >= 0).all()
                    maximum = saved["maximum"][...]
                    for index, tile in enumerate(coordinates):
                        first = tile * geometry["core"]
                        lengths = np.minimum(geometry["core"], geometry["shape"] - first)
                        if np.any(lengths <= 0):
                            continue
                        source_slices = tuple(slice(0, int(size)) for size in lengths)
                        target_slices = tuple(slice(int(begin), int(begin + size)) for begin, size in zip(first, lengths))
                        block = compressed[(index, slice(None), *source_slices)]
                        saved["values"][(slice(None), *target_slices)] = block
                        maximum = np.maximum(maximum, block.max((1, 2, 3)).astype(np.float32))
                        if smoke:
                            np.testing.assert_array_equal(saved["values"][(slice(None), *target_slices)], block)
                    saved["maximum"][...] = maximum
                    saved.flush()
                    saved.attrs["completed_tiles"] = stop
                    saved.flush()
                elapsed = time.monotonic() - started
                completed = offset + len(selected)
                rate = (completed - initial) / elapsed
                state = {"state": "running", "model": model, "case": case, "cases": len(audit["cases"]), "completed": completed, "total": len(indices), "tiles_per_second": rate, "case_eta_seconds": (len(indices) - completed) / rate, "updated_at": timestamp()}
                write_json(output_root() / f"progress_{model}.json", state)
                if offset % (batch_size * 8) == 0 or completed == len(indices):
                    print(state, flush=True)
                    progress(campaign_root(), task, "running", completed, len(indices), started, worker="local", case=case + 1, cases=len(audit["cases"]))
            for run, saved in outputs.items():
                assert saved.attrs["completed_tiles"] == len(indices)
                saved.attrs["status"] = "smoke_completed" if smoke else "completed"
                saved.flush()
                write_json(paths[run].with_suffix(".json"), {"state": saved.attrs["status"], "run_id": run, "case": case, "tiles": len(indices), "shape": geometries[run]["shape"].tolist(), "dictionary_sha256": saved.attrs["dictionary_sha256"], "seconds": time.monotonic() - started, "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20, "created_at": timestamp(), "test_used": False})
    write_json(output_root() / f"{'smoke_' if smoke else ''}completed_{model}.json", {"state": "completed", "runs": encoder.runs, "cases": 1 if smoke else len(audit["cases"]), "created_at": timestamp()})
    write_json(output_root() / f"progress_{model}.json", {"state": "smoke_completed" if smoke else "completed", "model": model, "cases": 1 if smoke else len(audit["cases"]), "updated_at": timestamp()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fmcib", "vista"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_model(args.model, args.smoke)

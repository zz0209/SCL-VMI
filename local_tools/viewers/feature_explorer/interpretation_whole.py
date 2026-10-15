import argparse
import atexit
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch
from filelock import FileLock
from scipy.ndimage import affine_transform, map_coordinates

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json
from sclvmi.models import load_encoder
from sclvmi.sae import SpatialDictionary
from sclvmi.sae_encoders import captured_sites, fmcib_prefix, saved_random_encoder
from whole import input_affine, load_case


CONFIG = read_json(ROOT / "configs/sae_interpretation.json")
SETTINGS = CONFIG["whole_ct"]
REGIONS = ["all", "lung", "liver", "other_tissue", "high_hu", "edge_air"]
INTENSITY = {"fmcib": (-1024.0, 2048.0), "vista": (-963.8247715525971, 1053.678477684517)}
MODEL_SITES = {"fmcib": ["layer1_context", "layer2"], "vista": ["stage2", "stage3"]}
_outputs = {}


def runs_root():
    return Path(context()[0]["runs"])


def output_root():
    return runs_root() / SETTINGS["run_id"]


def audit():
    return read_json(runs_root() / SETTINGS["source_run"] / "source_audit.json")


def dictionaries(model, site, trained_seeds=None, random_seeds=None):
    choice = CONFIG["production"]["sites"][f"{model}:{site}"]
    name = f"{CONFIG['campaign']}_{{}}_{model}_{site}_e{choice['expansion']}_k{choice['k']}_s{{}}_{choice['variant']}"
    trained = SETTINGS["trained_seeds"] if trained_seeds is None else trained_seeds
    random = SETTINGS["random_seeds"] if random_seeds is None else random_seeds
    return [("trained", name.format("trained", seed)) for seed in trained] + [("random", name.format("random", seed)) for seed in random]


def window_affine(model, start, offset):
    shift = np.eye(4)
    shift[:3, 3] = offset
    return input_affine(model, start) @ shift


def window_input(volume, affine, model, start, size, offset):
    transform = np.linalg.inv(affine) @ window_affine(model, start, offset)
    low, high = INTENSITY[model]
    crop = affine_transform(volume, transform[:3, :3], offset=transform[:3, 3], output_shape=(size,) * 3, order=1, mode="constant", cval=low, prefilter=False)
    crop = (crop - low) / (high - low)
    if model == "vista":
        np.clip(crop, 0, 1, out=crop)
    return crop[None].astype(np.float32)


def geometry(model, site, shape, affine):
    spec = SETTINGS["sites"][f"{model}:{site}"]
    core, step = spec["count"], spec["step_mm"]
    spacing = step / core
    extent = (np.asarray(shape) - 1) * nib.affines.voxel_sizes(affine)
    origin = affine[:3, 3].copy()
    shift = np.eye(4)
    shift[:3, 3] = spec["first"]
    native_affine = window_affine(model, origin, spec["window_offset"]) @ np.diag([spec["stride"]] * 3 + [1]) @ shift
    transform = nib.orientations.ornt_transform(nib.orientations.io_orientation(native_affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
    canonical_affine = native_affine @ nib.orientations.inv_ornt_aff(transform, (core,) * 3)
    assert np.allclose(canonical_affine[:3, :3], np.eye(3) * spacing)
    native_origin = canonical_affine[:3, 3]
    native_shape = np.floor((origin + extent - native_origin) / spacing).astype(int) + 1
    tiles = -(-native_shape // core)
    assert np.all(native_shape > 0)
    return {"tiles": tiles, "shape": native_shape, "origin": native_origin, "spacing": spacing, "core": core, "tile_origin": origin, "tile_step": step, "transform": transform, "extent": extent}


def orient(values, transform):
    flips = [2 + axis for axis in range(3) if transform[axis, 1] == -1]
    if flips:
        values = values.flip(flips)
    order = [0, 1] + [2 + int(np.flatnonzero(transform[:, 0] == axis)[0]) for axis in range(3)]
    return values.permute(order)


def valid_positions(geometry_value, receptive_field):
    half = receptive_field // 2
    coordinates = [geometry_value["origin"][axis] + np.arange(geometry_value["shape"][axis]) * geometry_value["spacing"] for axis in range(3)]
    low, high = geometry_value["tile_origin"], geometry_value["tile_origin"] + geometry_value["extent"]
    axes = [(values - half >= low[axis]) & (values + half <= high[axis]) for axis, values in enumerate(coordinates)]
    return axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]


class Windows(torch.utils.data.Dataset):
    def __init__(self, volume, affine, model, starts, size, offset):
        self.volume, self.affine, self.model, self.starts, self.size, self.offset = volume, affine, model, starts, size, offset

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        torch.set_num_threads(1)
        return torch.from_numpy(window_input(self.volume, self.affine, self.model, self.starts[index], self.size, self.offset))


class Encoders:
    def __init__(self, model, random=True):
        self.model = model
        self.trained = load_encoder(model)[0].cuda()
        self.random = None
        if random:
            path = runs_root() / CONFIG["campaign"] / "random_encoders" / f"{model}.pt"
            self.random = saved_random_encoder(model, path, SETTINGS["random_encoder_sha256"][model]).cuda()

    @torch.inference_mode()
    def features(self, encoder, batch):
        network = self.trained if encoder == "trained" else self.random
        if self.model == "fmcib":
            return {"layer1_context": fmcib_prefix(network, batch, "layer1")} if batch.shape[-1] == SETTINGS["sites"]["fmcib:layer1_context"]["window"] else {"layer2": fmcib_prefix(network, batch, "layer2")}
        return captured_sites(network, "vista", batch, {site: CONFIG["models"]["vista"][site]["module"] for site in MODEL_SITES["vista"]})


def open_outputs(paths):
    torch.set_num_threads(1)
    for run, path in paths.items():
        _outputs[run] = h5py.File(path, "r+")
    atexit.register(close_outputs)


def close_outputs():
    for saved in _outputs.values():
        saved.close()
    _outputs.clear()


def write_batch(blocks, coordinates, offset):
    stop = offset + len(coordinates)
    for run, compressed in blocks.items():
        saved = _outputs[run]
        if saved.attrs["completed_tiles"] >= stop:
            continue
        assert saved.attrs["completed_tiles"] == offset
        core = int(saved.attrs["core_size"])
        shape = np.asarray(saved.attrs["shape"])
        maximum = saved["maximum"][...]
        for index, tile in enumerate(coordinates):
            first = tile * core
            lengths = np.minimum(core, shape - first)
            source = tuple(slice(0, int(length)) for length in lengths)
            target = tuple(slice(int(begin), int(begin + length)) for begin, length in zip(first, lengths))
            block = compressed[(index, slice(None), *source)]
            if "valid" in saved:
                block = block * saved["valid"][target][None]
            saved["values"][(slice(None), *target)] = block
            maximum = np.maximum(maximum, block.max((1, 2, 3)).astype(np.float32))
        saved["maximum"][...] = maximum
        saved.flush()
        saved.attrs["completed_tiles"] = stop
        saved.flush()
    return stop


def initialize(path, run, value, count, case, valid):
    with h5py.File(path, "a") as saved:
        if "values" in saved:
            assert saved.attrs["run_id"] == run and saved.attrs["total_tiles"] == count
            return int(saved.attrs["completed_tiles"])
        request = read_json(runs_root() / CONFIG["campaign"] / "runs" / run / "request.json")
        core = value["core"]
        saved.create_dataset("values", (request["dict_size"], *value["shape"]), dtype="f2", chunks=(min(32, request["dict_size"]), *np.minimum(core, value["shape"])), compression="lzf", shuffle=True)
        saved.create_dataset("maximum", data=np.zeros(request["dict_size"], np.float32))
        if valid is not None:
            saved.create_dataset("valid", data=valid.astype(np.uint8), compression="gzip", compression_opts=1)
        saved.attrs.update(run_id=run, completed_tiles=0, total_tiles=count, status="running", case_index=case, shape=value["shape"], tiles=value["tiles"], origin=value["origin"], step_mm=value["spacing"], core_size=core, tile_origin=value["tile_origin"], tile_step_mm=value["tile_step"], dictionary_sha256=sha256(runs_root() / CONFIG["campaign"] / "runs" / run / "dictionary.pt"))
        return 0


def maps(model, sites_filter, trained_seeds, random_seeds, batch_size, writers, workers, cases, limit):
    root = output_root() / ("smoke_maps" if limit else "maps")
    root.mkdir(parents=True, exist_ok=True)
    identity = {"configuration": SETTINGS, "production": CONFIG["production"], "source_audit_sha256": sha256(runs_root() / SETTINGS["source_run"] / "source_audit.json"), "limit": limit, "test_used": False}
    code = {"code_sha256": sha256(Path(__file__)), "window_code_sha256": sha256(Path(__file__).with_name("whole.py")), "encoder_code_sha256": sha256(ROOT / "src/sclvmi/sae_encoders.py")}
    request = root / "request.json"
    if request.exists():
        assert read_json(request) == identity, "Changed whole-CT request requires a new output directory"
    else:
        write_json(request, identity)
    encoders = Encoders(model, random=bool(random_seeds))
    selected = {site: dictionaries(model, site, trained_seeds, random_seeds) for site in MODEL_SITES[model] if site in sites_filter}
    sae = {run: SpatialDictionary(runs_root() / CONFIG["campaign"] / "runs" / run / "dictionary.pt") for runs in selected.values() for _, run in runs}
    passes = [[site] for site in ["layer1_context", "layer2"] if site in selected] if model == "fmcib" else [[site for site in ["stage2", "stage3"] if site in selected]]
    rows = audit()["cases"]
    for case in cases:
        volume, affine = load_case(rows[case])
        for sites in passes:
            spec = SETTINGS["sites"][f"{model}:{sites[0]}"]
            geometries = {site: geometry(model, site, volume.shape, affine) for site in sites}
            tiles = geometries[sites[0]]["tiles"]
            for site in sites[1:]:
                assert np.array_equal(geometries[site]["tiles"], tiles) and geometries[site]["tile_step"] == spec["step_mm"]
            indices = np.arange(int(np.prod(tiles)))
            if limit:
                center = np.ravel_multi_index(tuple(tiles // 2), tiles)
                indices = indices[center:center + limit]
            jobs = [(encoder, run, site) for site in sites for encoder, run in selected[site]]
            encoders_needed = sorted({encoder for encoder, _, _ in jobs})
            paths, positions = {}, []
            for encoder, run, site in jobs:
                path = root / run / f"case_{case:02d}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                valid = valid_positions(geometries[site], SETTINGS["sites"][f"{model}:{site}"]["receptive_field"]) if "receptive_field" in SETTINGS["sites"][f"{model}:{site}"] else None
                positions.append(initialize(path, run, geometries[site], len(indices), case, valid))
                paths[run] = str(path)
            completed = min(positions)
            if completed == len(indices):
                continue
            starts = np.asarray(np.unravel_index(indices[completed:], tiles)).T * spec["step_mm"] + affine[:3, 3]
            loader = torch.utils.data.DataLoader(Windows(volume, affine, model, starts, spec["window"], spec["window_offset"]), batch_size=batch_size, num_workers=workers, persistent_workers=False, prefetch_factor=4)
            started, initial = time.perf_counter(), completed
            groups = [jobs[index::writers] for index in range(writers)]
            with ExitStack() as stack:
                pools = [stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=open_outputs, initargs=({run: paths[run] for _, run, _ in group},))) for group in groups if group]
                groups = [group for group in groups if group]
                pending = []

                def finish_pending():
                    nonlocal completed
                    if not pending:
                        return
                    stops = [future.result() for future in pending]
                    assert len(set(stops)) == 1
                    completed = stops[0]
                    pending.clear()
                    rate = (completed - initial) / (time.perf_counter() - started)
                    progress = {"state": "running", "model": model, "sites": sites, "case": case, "completed": completed, "total": len(indices), "windows_per_second": rate, "case_eta_seconds": (len(indices) - completed) / rate, "updated_at": timestamp()}
                    write_json(output_root() / f"progress_{model}.json", progress)

                offset = completed
                for batch in loader:
                    batch = batch.cuda(non_blocking=True)
                    coordinates = np.asarray(np.unravel_index(indices[offset:offset + len(batch)], tiles)).T
                    blocks = {}
                    with torch.inference_mode():
                        for encoder in encoders_needed:
                            values = encoders.features(encoder, batch)
                            for site in sites:
                                spec_site = SETTINGS["sites"][f"{model}:{site}"]
                                first, stop = spec_site["first"], spec_site["first"] + spec_site["count"]
                                core = values[site][:, :, first:stop, first:stop, first:stop]
                                tokens = core.permute(0, 2, 3, 4, 1).reshape(-1, core.shape[1])
                                for job_encoder, run, job_site in jobs:
                                    if job_encoder != encoder or job_site != site:
                                        continue
                                    encoded = sae[run].encode(tokens).reshape(len(batch), *core.shape[2:], -1).permute(0, 4, 1, 2, 3)
                                    encoded = orient(encoded, geometries[site]["transform"]).to(torch.float16).contiguous()
                                    assert torch.isfinite(encoded).all() and (encoded >= 0).all()
                                    blocks[run] = encoded.cpu().numpy()
                    finish_pending()
                    pending = [pool.submit(write_batch, {run: blocks[run] for _, run, _ in group}, coordinates, offset) for pool, group in zip(pools, groups)]
                    offset += len(batch)
                finish_pending()
                for pool in pools:
                    pool.submit(close_outputs).result()
            assert completed == len(indices)
            elapsed = time.perf_counter() - started
            for _, run, _ in jobs:
                with h5py.File(paths[run], "r+") as saved:
                    assert saved.attrs["completed_tiles"] == len(indices)
                    saved.attrs["status"] = "smoke_completed" if limit else "completed"
                    receipt = {"state": str(saved.attrs["status"]), "run_id": run, "case": case, "tiles": len(indices), "shape": [int(value) for value in saved.attrs["shape"]], "dictionary_sha256": str(saved.attrs["dictionary_sha256"]), "seconds": elapsed, "windows_per_second": (len(indices) - initial) / elapsed, "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20, **code, "created_at": timestamp(), "test_used": False}
                write_json(Path(paths[run]).with_suffix(".json"), receipt)
            print({"model": model, "sites": sites, "case": case, "windows": len(indices) - initial, "seconds": round(elapsed, 1), "windows_per_second": round((len(indices) - initial) / elapsed, 2)}, flush=True)
    if not limit:
        write_json(output_root() / "executions" / f"maps_{model}_{'_'.join(sites_filter)}_{timestamp().replace(':', '').replace('+', '_')}.json", {"state": "completed", "cases": list(cases), "runs": [run for runs in selected.values() for _, run in runs], "created_at": timestamp()})


def region_masks(row, value):
    image = nib.as_closest_canonical(nib.load(row["image"]))
    volume, affine = image.get_fdata(dtype=np.float32), image.affine
    lung_image, liver_image = nib.load(row["lung_mask"]), nib.load(row["liver_mask"])
    indices = np.indices(tuple(value["shape"])).reshape(3, -1).T
    points = value["origin"] + indices * value["spacing"]
    source = nib.affines.apply_affine(np.linalg.inv(affine), points)
    inside = ((source >= 0) & (source <= np.asarray(volume.shape) - 1)).all(-1)
    hu = map_coordinates(volume, source.T, order=1, mode="constant", cval=-1024, prefilter=False)
    lung = map_coordinates(np.asarray(lung_image.dataobj, dtype=np.uint8), nib.affines.apply_affine(np.linalg.inv(lung_image.affine), points).T, order=0, mode="constant", cval=0, prefilter=False) > 0
    liver = map_coordinates(np.asarray(liver_image.dataobj, dtype=np.uint8), nib.affines.apply_affine(np.linalg.inv(liver_image.affine), points).T, order=0, mode="constant", cval=0, prefilter=False) > 0
    relative = source / (np.asarray(volume.shape) - 1)
    edge = ((relative[:, :2] < .1) | (relative[:, :2] > .9)).any(1) & (hu < -950)
    other = (hu > -500) & (hu < 300) & ~lung & ~liver
    masks = np.stack([inside, lung & inside, liver & inside, other & inside, (hu >= 300) & inside, edge & inside], axis=1)
    return masks.reshape(*value["shape"], len(REGIONS))


def cache_file(source, destination, row):
    target = destination / source.parent.name / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix(".json")
    with FileLock(str(target.with_suffix(".conversion.lock"))):
        if receipt.exists():
            return read_json(receipt)
        return convert(source, target, receipt, row)


def convert(source, target, receipt, row):
    identity = {"source_bytes": source.stat().st_size, "source_receipt_sha256": sha256(source.with_suffix(".json")), "generator_sha256": sha256(Path(__file__))}
    started = time.perf_counter()
    partial = target.with_suffix(".partial.h5")
    partial.unlink(missing_ok=True)
    with h5py.File(source, "r", driver="core", backing_store=False) as saved, h5py.File(partial, "w") as output:
        assert saved.attrs["status"] in ["completed", "smoke_completed"]
        shape = saved["values"].shape
        value = {"shape": np.asarray(saved.attrs["shape"]), "origin": np.asarray(saved.attrs["origin"]), "spacing": float(saved.attrs["step_mm"])}
        masks = region_masks(row, value)
        if "valid" in saved:
            masks &= saved["valid"][...].astype(bool)[..., None]
            output.create_dataset("valid", data=saved["valid"][...], compression="gzip", compression_opts=1)
        flat_masks = masks.reshape(-1, len(REGIONS)).astype(np.float64)
        for key, item in saved.attrs.items():
            output.attrs[key] = item
        for key, item in identity.items():
            output.attrs[key] = item
        output.attrs["regions"] = REGIONS
        output.create_dataset("values", shape, dtype="float16", chunks=(1, *shape[1:]), compression="gzip", compression_opts=1, shuffle=True)
        output.create_dataset("maximum", data=saved["maximum"][...])
        sums = np.zeros((shape[0], len(REGIONS)))
        step = saved["values"].chunks[0]
        for first in range(0, shape[0], step):
            values = saved["values"][first:first + step]
            assert np.isfinite(values).all() and (values >= 0).all()
            output["values"][first:first + step] = values
            np.testing.assert_array_equal(output["values"][first:first + step], values)
            for index, feature in enumerate(values):
                flat = feature.reshape(-1)
                nonzero = np.flatnonzero(flat)
                sums[first + index] = flat[nonzero].astype(np.float64) @ flat_masks[nonzero]
        output.create_dataset("region_sums", data=sums)
        output.create_dataset("region_counts", data=flat_masks.sum(0))
        output.attrs["status"] = str(saved.attrs["status"])
    partial.replace(target)
    result = {"state": str(read_json(source.with_suffix(".json"))["state"]), "identity": identity, "run_id": source.parent.name, "case": int(read_json(source.with_suffix(".json"))["case"]), "shape": list(shape), "bytes": target.stat().st_size, "sha256": sha256(target), "seconds": time.perf_counter() - started, "verification": "Every copied float16 value compared with the map", "created_at": timestamp(), "test_used": False}
    write_json(receipt, result)
    print({key: result[key] for key in ["run_id", "case", "bytes", "seconds"]}, flush=True)
    return result


def caches(workers, smoke, runs):
    root = output_root()
    source_root = root / ("smoke_maps" if smoke else "maps")
    destination = root / ("smoke_cache" if smoke else "browser_cache")
    rows = audit()["cases"]
    order = lambda path: (f"_s{SETTINGS['trained_seeds'][0]}_" not in path.parent.name or "_random_" in path.parent.name, path.parent.name, path.name)
    sources = sorted((path for path in source_root.glob("*/case_*.h5") if path.with_suffix(".json").exists() and (not runs or path.parent.name in runs)), key=order)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(cache_file, sources, [destination] * len(sources), [rows[int(read_json(path.with_suffix(".json"))["case"])] for path in sources]))
    write_json(destination / "completed.json", {"state": "completed", "files": len(results), "bytes": sum(row["bytes"] for row in results), "generator_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["maps", "caches"], required=True)
    parser.add_argument("--model", choices=["fmcib", "vista"])
    parser.add_argument("--sites", default="layer1_context,layer2,stage2,stage3")
    parser.add_argument("--trained-seeds", default=",".join(str(seed) for seed in SETTINGS["trained_seeds"]))
    parser.add_argument("--random-seeds", default=",".join(str(seed) for seed in SETTINGS["random_seeds"]))
    parser.add_argument("--cases", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=SETTINGS["batch_size"])
    parser.add_argument("--writers", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="windows per case and pass for a smoke run")
    parser.add_argument("--runs", default="", help="cache stage: convert only these run IDs")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage == "maps":
        seeds = lambda text: [int(value) for value in text.split(",") if value]
        maps(args.model, args.sites.split(","), seeds(args.trained_seeds), seeds(args.random_seeds), args.batch_size, args.writers, args.workers, [int(value) for value in args.cases.split(",")], args.limit)
    else:
        caches(args.workers, bool(args.limit), [value for value in args.runs.split(",") if value])


if __name__ == "__main__":
    main()

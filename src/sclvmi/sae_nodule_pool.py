import argparse
import hashlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae_interp_profiles import SITE_KEYS


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
BANDS = [("inside", -np.inf, 0.0), ("margin_0_4mm", 0.0, 4.0), ("near_4_10mm", 4.0, 10.0)]
ENCODERS = ["trained", "random"]


def pool_directory(root, encoder, model, site):
    return root / "nodule_pools" / encoder / model / site


def read_record(root, previous, model, sites, identifier):
    # Whole-file reads are sequential on the external drive; per-member reads seek within each archive.
    sources = {"new": np.load(io.BytesIO((root / "cases" / model / f"{identifier}.npz").read_bytes())), "previous": np.load(io.BytesIO((previous / model / "cases" / f"{identifier}.npz").read_bytes()))}
    result = {}
    for site in sites:
        keys = SITE_KEYS[(model, site)]
        distance = sources["new"][keys["distance"]]
        valid = sources["new"][keys["valid"]] if keys["valid"] else np.ones(distance.shape, dtype=bool)
        result[site] = (sources[keys["trained"][0]][keys["trained"][1]], sources[keys["random"][0]][keys["random"][1]], distance, valid)
    return result


def write_pool(root, encoder, model, site, pools, lengths, started):
    destination = pool_directory(root, encoder, model, site)
    destination.mkdir(parents=True, exist_ok=True)
    matrix = read_json(root / ("matrices" if encoder == "trained" else "matrices_random") / model / site / "matrix.json")
    files, counts = {}, {}
    for name, _, _ in BANDS:
        values = np.concatenate(pools[name])
        offsets = np.concatenate([[0], np.cumsum(lengths[name])]).astype(np.int64)
        assert offsets[-1] == len(values) and values.shape[1] == matrix["channels"]
        np.save(destination / f"band_{name}.npy", values)
        np.save(destination / f"band_{name}_offsets.npy", offsets)
        for path in [destination / f"band_{name}.npy", destination / f"band_{name}_offsets.npy"]:
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        counts[name] = {"tokens": int(len(values)), "patients": int(len(offsets) - 1)}
    identity = hashlib.sha256(("".join(name + info["sha256"] for name, info in sorted(files.items())) + matrix["identity"]).encode()).hexdigest()
    write_json(destination / "pool.json", {"encoder": encoder, "model": model, "site": site, "bands": [{"name": name, "low_mm": low, "high_mm": high} for name, low, high in BANDS],
                                           "rule": "all valid positions of every training record whose signed distance to the target nodule surface lies in the band; one segment per training patient with at least one position",
                                           "counts": counts, "files": files, "matrix_identity": matrix["identity"], "identity": identity, "code_sha256": sha256(Path(__file__)),
                                           "seconds": time.monotonic() - started, "created_at": timestamp(), "test_used": False})
    print("POOL", encoder, model, site, counts, flush=True)


def build(root, previous, model, sites):
    members = pd.read_csv(root / "members.csv", dtype={"PatientID": str})
    members = members[members.split == "train"].sort_values(["PatientID", "identifier"])
    pools = {(encoder, site): {name: [] for name, _, _ in BANDS} for encoder in ENCODERS for site in sites}
    lengths = {key: {name: [] for name, _, _ in BANDS} for key in pools}
    started = time.monotonic()
    patients = members.groupby("PatientID", sort=True)
    for index, (_, records) in enumerate(patients):
        collected = {key: {name: [] for name, _, _ in BANDS} for key in pools}
        for identifier in records.identifier:
            for site, (trained, random, distance, valid) in read_record(root, previous, model, sites, identifier).items():
                flat_valid = valid.reshape(-1)
                flat_distance = distance.reshape(-1)[flat_valid]
                tokens = {"trained": trained.reshape(trained.shape[0], -1).T[flat_valid], "random": random.reshape(random.shape[0], -1).T[flat_valid]}
                for name, low, high in BANDS:
                    mask = (flat_distance >= low) & (flat_distance < high)
                    for encoder in ENCODERS:
                        collected[(encoder, site)][name].append(tokens[encoder][mask])
        for key, bands in collected.items():
            for name, parts in bands.items():
                values = np.concatenate(parts).astype(np.float32)
                if len(values):
                    pools[key][name].append(values)
                    lengths[key][name].append(len(values))
        if (index + 1) % 100 == 0 or index + 1 == patients.ngroups:
            print(f"{model}:{','.join(sites)} pools {index + 1}/{patients.ngroups} patients, {time.monotonic() - started:.0f} s", flush=True)
    for encoder, site in pools:
        write_pool(root, encoder, model, site, pools[(encoder, site)], lengths[(encoder, site)], started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="fmcib:layer2,vista:stage2+stage3", help="comma-separated model:site+site groups; each group reads every record once")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    for group in args.models.split(","):
        model, sites = group.split(":")
        build(root, Path(storage["activations"]) / config["previous_suite"], model, sites.split("+"))


if __name__ == "__main__":
    main()

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae import SpatialDictionary
from .sae_campaign_evaluate import metrics
from .sae_sites import SpatialEncoder, volume_with_geometry


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
INTERIOR = (slice(None), slice(None), slice(5, 9), slice(5, 9), slice(5, 9))


def reconstruct(dictionary, features):
    channels, *shape = features.shape[1:]
    tokens = features.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    recovered = dictionary.reconstruct(tokens, batch=16384)
    return recovered.reshape(features.shape[0], *shape, channels).permute(0, 4, 1, 2, 3).contiguous()


@torch.inference_mode()
def raw_check(encoder, dictionary, deployed, row, expected):
    tensor, _ = volume_with_geometry(row.image, encoder.name)
    captured = encoder.extract(tensor, sites=[deployed])
    recovered = reconstruct(dictionary, captured[deployed])
    final = encoder.replace(tensor, deployed, recovered)
    return abs(float(encoder.probability(final)[0]) - expected)


def record(row, original, recovered):
    return {"identifier": row.identifier, "PatientID": row.PatientID, "label": row.label, "original": original, "recovered": float(recovered)}


@torch.inference_mode()
def evaluate(root, model, runs_by_site, batch_size=8):
    config = read_json(CONFIG_PATH)
    deployed = {site: config["models"][model][site]["deployed_site"] for site in runs_by_site}
    storage, _, _ = context()
    previous = Path(storage["activations"]) / config["previous_suite"] / model
    members = pd.read_csv(previous / "full_members.csv", dtype={"PatientID": str}).query("split == 'development'").sort_values("identifier").reset_index(drop=True)
    encoder = SpatialEncoder(model)
    dictionaries = {(site, run): SpatialDictionary(root / "runs" / run / "dictionary.pt") for site, runs in runs_by_site.items() for run in runs}
    for (site, run), dictionary in dictionaries.items():
        assert dictionary.request["model"] == model and dictionary.request["site"] == site, run
    rows = {key: [] for key in dictionaries}
    interior_rows = {key: [] for key in dictionaries if key[0] == "layer1_context"}
    started = time.monotonic()
    for start in range(0, len(members), batch_size):
        batch = members.iloc[start:start + batch_size]
        features = {name: [] for name in set(deployed.values())}
        saved_probability = []
        for row in batch.itertuples():
            with np.load(io.BytesIO((previous / "cases" / f"{row.identifier}.npz").read_bytes())) as saved:
                for name in features:
                    features[name].append(torch.from_numpy(saved[name]))
                saved_probability.append(float(saved["probability"][0]))
        features = {name: torch.stack(values).cuda() for name, values in features.items()}
        for name, values in features.items():
            np.testing.assert_allclose(encoder.probability(encoder.continue_from(name, values)), saved_probability, rtol=0, atol=1e-5)
        for (site, run), dictionary in dictionaries.items():
            values = features[deployed[site]]
            recovered = reconstruct(dictionary, values)
            probability = encoder.probability(encoder.continue_from(deployed[site], recovered))
            rows[(site, run)].extend(record(row, original, new) for row, original, new in zip(batch.itertuples(), saved_probability, probability))
            if site == "layer1_context":
                partial = values.clone()
                partial[INTERIOR] = recovered[INTERIOR]
                probability = encoder.probability(encoder.continue_from(deployed[site], partial))
                interior_rows[(site, run)].extend(record(row, original, new) for row, original, new in zip(batch.itertuples(), saved_probability, probability))
        if start % (batch_size * 20) == 0:
            write_json(root / "progress" / f"evaluate-{model}.json", {"task": "evaluate", "completed": start + len(batch), "total": len(members), "dictionaries": len(dictionaries), "elapsed_seconds": time.monotonic() - started, "updated_at": timestamp()})
    limits = config["acceptance"]
    for (site, run), dictionary in dictionaries.items():
        frame = pd.DataFrame(rows[(site, run)])
        destination = root / "runs" / run / "downstream"
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "predictions.csv", index=False)
        values = metrics(frame)
        checks = {"probability_mae": values["probability_mae"] <= limits["max_probability_mae"], "auroc": values["original_auroc"] - values["recovered_auroc"] <= limits["max_auroc_decrease"], "ap": values["original_ap"] - values["recovered_ap"] <= limits["max_ap_decrease"]}
        raw = []
        for row in members.groupby("label", sort=True).head(1).itertuples():
            expected = frame.loc[frame.identifier == row.identifier, "recovered"].item()
            error = raw_check(encoder, dictionary, deployed[site], row, expected)
            # Raw CT path re-extracts the deployed map; float32 differences can flip a feature at its threshold.
            assert error < 1e-4, (run, row.identifier, error)
            raw.append({"identifier": row.identifier, "absolute_probability_error": error})
        result = {"state": "completed", "run_id": run, "replacement": f"full deployed {deployed[site]} map", "metrics": values, "checks": checks, "checks_pass": all(checks.values()), "raw_checks": raw, "dictionary_sha256": sha256(root / "runs" / run / "dictionary.pt"), "code_sha256": sha256(Path(__file__)), "members": len(members), "elapsed_seconds": time.monotonic() - started, "completed_at": timestamp(), "test_used": False}
        if (site, run) in interior_rows:
            partial = pd.DataFrame(interior_rows[(site, run)])
            partial.to_csv(destination / "interior_predictions.csv", index=False)
            result["interior_replacement"] = {"positions": "deployed layer1 native indices 5-8 on each axis, the complete receptive-field positions", "metrics": metrics(partial)}
        write_json(root / "runs" / run / "downstream.json", result)
        print("EVALUATED", run, values["probability_mae"], values["recovered_auroc"], result["checks_pass"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--seeds", default="2025")
    parser.add_argument("--groups")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant", help="evaluate only runs of this training variant; an empty value selects the 12,000-step grid")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = read_json(CONFIG_PATH)
    storage, _, _ = context()
    root = Path(storage["runs"]) / config["campaign"]
    seeds = {int(value) for value in args.seeds.split(",")}
    groups = set(args.groups.split(",")) if args.groups else None
    runs_by_site = {}
    for site in args.site.split(","):
        for path in sorted((root / "runs").glob(f"{config['campaign']}_trained_{args.model}_{site}_e*")):
            request = read_json(path / "request.json") if (path / "request.json").exists() else None
            if request is None or request["seed"] not in seeds or not (path / "dictionary.pt").exists() or (path / "downstream.json").exists():
                continue
            if groups and f"e{request['expansion']}_k{request['k']}" not in groups:
                continue
            if args.variant is not None and not path.name.endswith(f"_s{request['seed']}" + (f"_{args.variant}" if args.variant else "")):
                continue
            runs_by_site.setdefault(site, []).append(path.name)
    print("EVALUATING", runs_by_site, flush=True)
    if runs_by_site:
        evaluate(root, args.model, runs_by_site, args.batch_size)


if __name__ == "__main__":
    main()

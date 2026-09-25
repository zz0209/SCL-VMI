import argparse
import gc
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from sclvmi.context import context, read_json, timestamp, write_json
from sclvmi.features import extract
from sclvmi.head import fit_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    storage, _, _ = context()
    output = Path(storage["runs"]) / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    with threadpool_limits(limits=4):
        for name in ["fmcib", "ctfm", "vista", "genesis"]:
            head_run = args.run_id + "_" + name
            result_path = Path(storage["runs"]) / head_run / "result.json"
            if result_path.exists():
                records[name] = read_json(result_path)
                continue
            print(f"Frozen baseline: {name}", flush=True)
            directory = extract(name, ["train", "development"], spatial=False)
            fit_head(directory, head_run)
            records[name] = read_json(result_path)
            write_json(output / "status.json", {"status": "running", "updated_at": timestamp(), "completed_models": list(records)})
            gc.collect()
            torch.cuda.empty_cache()
    write_json(output / "status.json", {"status": "completed", "completed_at": timestamp(), "models": records, "test_evaluated": False})


if __name__ == "__main__":
    main()

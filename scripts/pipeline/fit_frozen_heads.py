import argparse
import subprocess
import sys
from pathlib import Path

from sclvmi.context import context, read_json, timestamp, write_json
from sclvmi.frozen_features import MODELS, SUITE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--heads-only", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--features", type=Path)
    parser.add_argument("--field-mm", type=float)
    parser.add_argument("--variant")
    args = parser.parse_args()
    if args.worker:
        from sclvmi.frozen_heads import run_heads

        run_heads(args.models[0], args.features, smoke=args.smoke_only, variant=args.variant)
        return
    storage, _, _ = context()
    assert args.field_mm is None or args.variant is not None, "Field comparison requires a distinct variant name"
    root = Path(storage["runs"]) / SUITE
    status_root = root if args.variant is None else root.with_name(SUITE+"_"+args.variant)
    root.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        extra = [] if args.field_mm is None else ["--field-mm", str(args.field_mm)]
        variant = [] if args.variant is None else ["--variant", args.variant]
        key = name if args.field_mm is None else f"{name}_fov{args.field_mm:g}"
        if not args.heads_only:
            subprocess.run([sys.executable, "-u", "-m", "sclvmi.frozen_features", "--model", name, "--limit", "8"]+extra, check=True)
            directory = read_json(root / f"{key}_smoke.json")["directory"]
            subprocess.run([sys.executable, "-u", __file__, "--worker", "--models", name, "--features", directory, "--smoke-only"]+variant, check=True)
            if args.smoke_only:
                continue
            subprocess.run([sys.executable, "-u", "-m", "sclvmi.frozen_features", "--model", name]+extra, check=True)
        directory = read_json(root / f"{key}_features.json")["directory"]
        subprocess.run([sys.executable, "-u", __file__, "--worker", "--models", name, "--features", directory]+variant, check=True)
        write_json(status_root / f"{name}_complete.json", {"status": "completed", "model": name, "completed_at": timestamp()})
    write_json(status_root / "progress.json", {"status": "completed", "models": args.models, "smoke": args.smoke_only, "completed_at": timestamp()})


if __name__ == "__main__":
    main()

import argparse

from .data import build_manifest


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("manifest")
    features = commands.add_parser("extract")
    features.add_argument("--model", choices=["fmcib", "ctfm", "vista", "genesis"], required=True)
    features.add_argument("--splits", nargs="+", choices=["train", "development", "test"], default=["train", "development"])
    features.add_argument("--limit", type=int)
    features.add_argument("--spatial", action="store_true")
    head = commands.add_parser("head")
    head.add_argument("--features", required=True)
    head.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        build_manifest()
    elif args.command == "extract":
        from .features import extract
        extract(args.model, args.splits, args.limit, args.spatial)
    elif args.command == "head":
        from .head import fit_head
        fit_head(args.features, args.run_id)


if __name__ == "__main__":
    main()

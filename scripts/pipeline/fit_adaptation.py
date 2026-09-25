import argparse
from pathlib import Path

from sclvmi.adaptation import fit, load_data, tensor_cache
from sclvmi.context import ROOT, context, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fmcib", "ctfm", "vista", "genesis"], required=True)
    parser.add_argument("--arm", choices=["frozen_mlp", "last_block", "last_block_translation"], required=True)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/adaptation.json")
    recipe = {"arm": args.arm, "candidate": args.candidate, "seed": args.seed}
    if args.arm == "frozen_mlp":
        settings = config["mlp"]
        candidates = [(lr, wd) for lr in settings["learning_rates"] for wd in settings["weight_decays"]]
        assert 0 <= args.candidate < len(candidates)
        lr, decay = candidates[args.candidate]
        recipe.update(mode="frozen", head="mlp", hidden=settings["hidden_dim"], dropout=settings["dropout"], lr=lr,
                      weight_decay=decay, batch_size=settings["batch_size"], epochs=settings["epochs"], patience=settings["patience"])
    else:
        settings = config["finetune"]
        assert 0 <= args.candidate < len(settings["encoder_learning_rates"])
        recipe.update(mode="last_block", head="linear", lr=settings["encoder_learning_rates"][args.candidate],
                      head_lr_multiplier=settings["head_learning_rate_multiplier"], weight_decay=settings["weight_decay"],
                      batch_size=1 if args.model == "genesis" else settings["batch_size"],
                      effective_batch_size=settings["effective_batch_size"], epochs=settings["epochs"],
                      translation_mm=settings["translation_mm"] if args.arm == "last_block_translation" else 0)
    storage, _, _ = context()
    baseline = Path(storage["runs"]) / args.baseline
    directory, train, dev, train_x, dev_x = load_data(baseline)
    images = tensor_cache(args.model, directory, [train, dev]) if recipe["mode"] == "last_block" else None
    fit(args.model, baseline, Path(storage["runs"]) / args.run_id, recipe, train, dev, train_x, dev_x, images)


if __name__ == "__main__":
    main()

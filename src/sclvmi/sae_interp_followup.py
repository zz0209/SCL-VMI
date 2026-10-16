import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .context import ROOT, context, read_json, sha256, timestamp, write_json
from .sae_interp_candidates import correlate, p_values, partial, ranks
from .sae_interp_profiles import profile_members


CONFIG_PATH = ROOT / "configs/sae_interpretation.json"
PUBLIC_COLUMNS = ["site", "feature", "family", "reappearance", "seed_agreement", "partial_discovery", "partial_confirmation", "q_confirmation",
                  "partial_morphology_confirmation", "p_morphology_confirmation", "size_only_auroc", "size_feature_auroc", "delta_auroc", "delta_auroc_low", "delta_auroc_high",
                  "development_auroc", "nodule_enrichment", "enrichment_percentile", "region_inside", "region_margin_0_4mm", "region_near_4_10mm", "region_far_over_10mm",
                  "size_rho", "top_reference", "top_reference_partial_rho", "image_r2", "frequency", "whole_lung_enrichment", "whole_liver_enrichment", "whole_high_hu_share", "whole_edge_air_share"]


def run_sites(output):
    return [site.replace(":", "_") for site in read_json(output / "request.json")["sites"]]


def scores(columns, labels, train, development):
    design = np.column_stack(columns)
    mean, scale = design[train].mean(0), design[train].std(0)
    design = (design - mean) / np.where(scale > 0, scale, 1)
    return LogisticRegression(max_iter=5000).fit(design[train], labels[train]).decision_function(design[development])


def bootstrap_difference(target, combined, base, count, generator):
    differences = []
    for _ in range(count):
        sample = generator.integers(0, len(target), len(target))
        if target[sample].min() != target[sample].max():
            differences.append(roc_auc_score(target[sample], combined[sample]) - roc_auc_score(target[sample], base[sample]))
    return np.percentile(differences, [2.5, 97.5])


def followup(root, output, settings):
    members = profile_members(root)
    references = pd.read_csv(root / "nodule_references.csv").set_index("identifier").loc[members.identifier].reset_index()
    labels = members.label.to_numpy()
    train = (members.split == "train").to_numpy()
    development = ~train
    size = np.log(references.volume_mm3.to_numpy())
    measured = np.column_stack([size, references[settings["references"]].to_numpy()])
    base = scores([measured], labels, train, development)
    rows, columns = [], {}
    for site in run_sites(output):
        candidates = pd.read_csv(output / site / "candidates.csv")
        with np.load(output / site / "summaries.npz") as saved:
            values = saved["trained"]
        for row in candidates[candidates.delta_auroc_low > 0].itertuples():
            summary = values[:, row.feature]
            columns[f"{site}:{row.feature}"] = summary
            combined = scores([measured, np.log1p(summary)], labels, train, development)
            low, high = bootstrap_difference(labels[development], combined, base, settings["bootstrap"], np.random.default_rng(settings["bootstrap_seed"]))
            confirmation = partial(summary[development, None], labels[development], measured[development])[0]
            rows.append({"site": site, "feature": row.feature, "family": row.family, "seed_agreement": row.seed_agreement,
                         "partial_all_references_discovery": partial(summary[train, None], labels[train], measured[train])[0],
                         "partial_all_references_confirmation": confirmation,
                         "p_all_references_confirmation": p_values(np.asarray([confirmation]), development.sum(), measured.shape[1])[0],
                         "references_auroc": roc_auc_score(labels[development], base), "references_feature_auroc": roc_auc_score(labels[development], combined),
                         "delta_auroc_low": low, "delta_auroc_high": high})
    frame = pd.DataFrame(rows)
    frame["delta_auroc"] = frame.references_feature_auroc - frame.references_auroc
    matrix = np.column_stack(list(columns.values()))
    centered = ranks(matrix[train]) - ranks(matrix[train]).mean(0)
    correlation = pd.DataFrame(correlate(centered, centered), index=list(columns), columns=list(columns))
    single = {name: {"discovery": float(partial(references[name].to_numpy()[train, None], labels[train], size[train, None])[0]),
                     "confirmation": float(partial(references[name].to_numpy()[development, None], labels[development], size[development, None])[0])}
              for name in settings["references"]}
    return frame, correlation, single


def figure(sites, candidates, frame, destination, order):
    colors = {"trained": "#2a78d6", "random": "#eb6834"}
    ink, muted, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    names = {"fmcib_layer1_context": "FMCIB\nlayer1 context", "fmcib_layer2": "FMCIB\nlayer2", "vista_stage2": "VISTA3D\nstage2", "vista_stage3": "VISTA3D\nstage3"}
    short = {"fmcib_layer1_context": "FMCIB L1", "fmcib_layer2": "FMCIB L2", "vista_stage2": "VISTA3D S2", "vista_stage3": "VISTA3D S3"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted, "text.color": ink})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=surface, gridspec_kw={"width_ratios": [1, 1.5]})
    positions = np.arange(len(order))
    for offset, encoder, label in [(-0.2, "trained", "trained encoder"), (0.2, "random", "random-weight encoder")]:
        counts = [sites[site][encoder]["confirmed"] for site in order]
        left.bar(positions + offset, counts, width=0.38, color=colors[encoder], label=label, edgecolor=surface, linewidth=1)
        for position, count in zip(positions + offset, counts):
            left.text(position, count + 0.4, str(count), ha="center", va="bottom", fontsize=8, color=ink)
    left.set_xticks(positions, [names[site] for site in order])
    left.set_ylabel("confirmed features (reference seed)")
    left.set_title("Stable features with size-adjusted malignancy association\nselected on training patients, confirmed on development patients", fontsize=9, loc="left")
    left.legend(frameon=False, loc="upper right")
    merged = frame.merge(candidates[["site", "feature", "delta_auroc", "delta_auroc_low", "delta_auroc_high"]], on=["site", "feature"], suffixes=("_references", "_volume"))
    merged = merged.assign(order=merged.site.map(order.index)).sort_values(["order", "delta_auroc_volume"], ascending=[False, True]).reset_index(drop=True)
    rows = np.arange(len(merged))
    right.hlines(rows + 0.15, merged.delta_auroc_low_volume, merged.delta_auroc_high_volume, color=colors["trained"], linewidth=2)
    right.plot(merged.delta_auroc_volume, rows + 0.15, "o", color=colors["trained"], markersize=6, markeredgecolor=surface, markeredgewidth=1.5, label="added to log volume")
    right.hlines(rows - 0.15, merged.delta_auroc_low_references, merged.delta_auroc_high_references, color=muted, linewidth=2)
    right.plot(merged.delta_auroc_references, rows - 0.15, "s", color=muted, markersize=6, markeredgecolor=surface, markeredgewidth=1.5, label="added to log volume + 8 nodule measurements")
    right.axvline(0, color=muted, linewidth=0.8)
    right.set_yticks(rows, [f"{short[site]} {feature}" for site, feature in zip(merged.site, merged.feature)])
    right.set_xlabel("change in development AUROC (95% patient bootstrap interval)")
    right.set_title("Added discrimination of single features over measured nodule properties", fontsize=9, loc="left")
    right.legend(frameon=False, loc="lower right")
    for axis in [left, right]:
        axis.set_facecolor(surface)
        axis.grid(axis="x" if axis is right else "y", color=grid, linewidth=0.6)
        axis.set_axisbelow(True)
        for side in ["top", "right"]:
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(destination / f"candidates.{suffix}", dpi=180, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", help="public results directory for feature-level tables")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    settings = config["candidates"]
    runs = Path(context()[0]["runs"])
    root = runs / config["campaign"]
    output = runs / settings["run_id"]
    frame, correlation, single = followup(root, output, settings)
    destination = output / "followup"
    destination.mkdir(exist_ok=True)
    frame.to_csv(destination / "all_references.csv", index=False)
    correlation.to_csv(destination / "candidate_correlation_discovery.csv")
    summary = {"rule": "candidates whose size-adjusted bootstrap interval of added development AUROC lies above zero; covariates are log volume plus the configured references",
               "references_size_adjusted_partial": single, "candidates": len(frame), "code_sha256": sha256(Path(__file__)), "created_at": timestamp(), "test_used": False}
    write_json(destination / "summary.json", summary)
    if args.export:
        public = Path(args.export)
        public.mkdir(parents=True, exist_ok=True)
        order = run_sites(output)
        tables = {encoder: pd.concat([pd.read_csv(output / site / f"{encoder}_features.csv").assign(site=site) for site in order]) for encoder in ["trained", "random"]}
        tables["trained"][tables["trained"].confirmed][PUBLIC_COLUMNS].to_csv(public / "confirmed_trained.csv", index=False)
        tables["random"][tables["random"].confirmed][PUBLIC_COLUMNS].to_csv(public / "confirmed_random.csv", index=False)
        sites = {key.replace(":", "_"): value for key, value in read_json(output / "summary.json")["sites"].items()}
        figure(sites, tables["trained"], frame, public, order)
        pd.DataFrame([{"site": site, "encoder": encoder, "random_threshold": value["threshold"], **value[encoder]} for site, value in sites.items() for encoder in ["trained", "random"]]).to_csv(public / "site_summary.csv", index=False)
        frame.to_csv(public / "all_references.csv", index=False)
        correlation.round(4).to_csv(public / "candidate_correlation_discovery.csv")
        write_json(public / "provenance.json", {"run_id": settings["run_id"], "campaign": config["campaign"], "settings": settings, "references_size_adjusted_partial": single,
                                                "request": read_json(output / "request.json"), "followup_code_sha256": summary["code_sha256"], "created_at": timestamp(), "test_used": False})
    pd.set_option("display.width", 250)
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

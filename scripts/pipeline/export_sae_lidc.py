import argparse
import platform
import shutil
from pathlib import Path

import pandas as pd

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


README = """# Nodule SAE features compared with radiologist ratings on LIDC-IDRI

Run `20261019_sae_lidc` asks which stable features of the [nodule-interpretation dictionaries](../20261015_sae_nodule_coverage/README.md) (seed 2025; FMCIB layer2 `nodule75`, VISTA3D stage2 `nodule16x`) follow nodule characteristics rated by radiologists on an independent dataset, beyond nodule size and beyond the random-weight-encoder features. The rules were fixed in `configs/sae_lidc.json` before any LIDC activation was computed.

![Size-adjusted partial Spearman correlation with each rating](ratings.png)

## Data

- **LIDC-IDRI** CT scans with slice thickness <= 2.5 mm. Annotations come from the pylidc database; annotations are grouped into nodules with pylidc, and a nodule enters when 3 or 4 radiologists annotated it.
  - {nodules:,} nodules from {patients} patients. {excluded} scans were excluded: {reasons}.
  - One nodule per patient (most readers, then largest): {members} nodules, split by patient into {discovery} discovery and {confirmation} confirmation nodules.
- **Crops:** 128 x 128 x 64 native voxels around the 50 % consensus centroid, voxels outside the scan set to -1024 HU. This is the rule of the LUNA25 crops used for the dictionaries, verified voxel by voxel against the official LUNA25 scans.
- **Ratings:** median over the readers of subtlety, sphericity, margin, lobulation, spiculation, texture and malignancy; for calcification, the fraction of readers who rated any calcification pattern.

## Procedure

- **Activations:** the crops pass through the same deployed inputs, frozen encoders, random-weight encoders and grid distance maps as the LUNA25 interpretation campaign. Recomputing six LUNA25 records along this path reproduced the cached arrays exactly. Each feature's target is its mean activation within 4 mm of the nodule surface or 4-10 mm outside it, whichever region was chosen on LUNA25 training patients.
- **Association:** partial Spearman correlation, i.e. the Pearson correlation of the ranks of target and rating after both are residualised on the rank of log consensus volume.
  - Discovery: a feature-rating pair is selected when the feature is active (target above 0) in at least 30 nodules of each half and its |correlation| exceeds the 95th percentile of the random-weight stable features for that rating.
  - Confirmation: the same sign and Benjamini-Hochberg q < 0.05 within the selected pairs.
  - Random-weight features go through the same steps.
- **Cross-seed replication:** the matched feature in each of the 7 other dictionary seeds has the same sign and p < 0.05 on the confirmation half; at least 6 of 7 are required.
- **Agreement between radiologists:** one reader drawn at random against the median of the others, size-adjusted in the same way, averaged over 200 draws. A feature compared with the median of all readers can exceed this value, because the median is more reliable than one reader.
- **Clear feature:** a confirmed, cross-seed replicated association of an active feature whose classifier effect ranks among the 50 largest (fixed LUNA25 classifier, [radiomics run](../20261017_sae_radiomics/README.md)).

## Results

{site_table}

{rating_table}

- **Clear features:** {clear_counts}. See `clear_features.csv`.
- **Prespecified review-list features:** the four LUNA25 contact descriptions hold on LIDC with the stated sign (size-adjusted, all nodules): {hypotheses}.

![Review-list features on LIDC](prespecified.png)

## Files

- `site_summary.csv`: nodules, analysed features, selected and confirmed pairs per site and encoder.
- `rating_summary.csv`: per rating, confirmed pairs and the largest confirmed |correlation| for trained and random-weight features, with the agreement between radiologists.
- `associations.csv`: every stable feature and rating: activity, discovery and confirmation correlations, selection, confirmation, replication, bootstrap interval and classifier effect rank.
- `clear_features.csv`, `clear_summary.csv`: the clear feature-rating pairs and one row per clear feature.
- `prespecified.csv`: the eight review-list features against all ratings and contact measures, with Holm-adjusted p-values.
- `ceiling.csv`: agreement between radiologists per rating.
- `provenance.json`: settings, counts and source hashes.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/sae_lidc.json")
    storage = context()[0]
    analysis = Path(storage["runs"]) / config["run_id"] / "analysis"
    nodules = read_json(Path(storage["preprocessed"]) / "lidc" / config["run_id"] / "nodules.json")
    destination = args.output
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["associations.csv", "clear_features.csv", "clear_summary.csv", "prespecified.csv", "ceiling.csv", "ratings.png", "ratings.svg", "prespecified.png", "prespecified.svg"]:
        shutil.copyfile(analysis / name, destination / name)
    summary = read_json(analysis / "summary.json")
    associations = pd.read_csv(analysis / "associations.csv")
    ceiling = pd.read_csv(analysis / "ceiling.csv").set_index("rating")
    sites = []
    for site, item in summary["sites"].items():
        for encoder in ["trained", "random"]:
            sites.append({"site": site, "encoder": encoder, "nodules": item["nodules"], "stable_features": item["stable"][encoder], "analysed_features": item["active"][encoder],
                          "selected_pairs": item["selected_pairs"].get(encoder, 0), "confirmed_pairs": item["confirmed_pairs"].get(encoder, 0),
                          "confirmed_pairs_per_analysed_feature": round(item["confirmed_pairs_per_active_feature"][encoder], 3), "clear_features": len(item["clear_features"]) if encoder == "trained" else None})
    sites = pd.DataFrame(sites)
    sites.to_csv(destination / "site_summary.csv", index=False)
    ratings = []
    for (site, rating), group in associations.groupby(["site", "rating"], sort=False):
        row = {"site": site, "rating": rating, "radiologist_agreement": ceiling.loc[rating, "reader_vs_others_size_adjusted_rho"]}
        for encoder in ["trained", "random"]:
            confirmed = group[(group.encoder == encoder) & group.active & group.confirmed]
            row[f"{encoder}_confirmed"] = len(confirmed)
            row[f"{encoder}_largest_abs_rho"] = round(confirmed.confirmation_rho.abs().max(), 3) if len(confirmed) else None
        row["clear_pairs"] = int(group.clear.sum())
        ratings.append(row)
    ratings = pd.DataFrame(ratings)
    ratings.to_csv(destination / "rating_summary.csv", index=False)
    site_table = ["| Site | Encoder | Stable | Analysed | Selected pairs | Confirmed pairs | Confirmed per analysed feature | Clear features |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in sites.itertuples():
        site_table.append(f"| {row.site} | {row.encoder} | {row.stable_features:,} | {row.analysed_features} | {row.selected_pairs} | {row.confirmed_pairs} | {row.confirmed_pairs_per_analysed_feature:.2f} | {'' if pd.isna(row.clear_features) else int(row.clear_features)} |")
    rating_table = ["| Site | Rating | Agreement between radiologists | Trained confirmed | Largest trained | Random confirmed | Largest random | Clear pairs |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in ratings.itertuples():
        largest = lambda value: "" if pd.isna(value) else f"{value:.2f}"
        rating_table.append(f"| {row.site} | {row.rating} | {row.radiologist_agreement:.2f} | {row.trained_confirmed} | {largest(row.trained_largest_abs_rho)} | {row.random_confirmed} | {largest(row.random_largest_abs_rho)} | {row.clear_pairs} |")
    prespecified = pd.read_csv(analysis / "prespecified.csv")
    stated = prespecified[prespecified.hypothesis_sign.notna()]
    readable = lambda measure: measure.replace("0_4mm", "0-4 mm").replace("4_10mm", "4-10 mm").replace("_", " ")
    hypotheses = "; ".join(f"{row.site} {row.feature} with {readable(row.measure)} {row.rho:+.2f} (Holm p {row.p_holm:.1e})" for row in stated.itertuples())
    clear_counts = " and ".join(f"{len(item['clear_features'])} at {site}" for site, item in summary["sites"].items())
    text = README.format(nodules=nodules["nodules"], patients=nodules["patients"], excluded=nodules["scans_excluded"], reasons=", ".join(f"{count} {reason}" for reason, count in nodules["exclusion_reasons"].items()),
                         members=nodules["members"], discovery=nodules["halves"]["discovery"], confirmation=nodules["halves"]["confirmation"],
                         site_table="\n".join(site_table), rating_table="\n".join(rating_table), clear_counts=clear_counts, hypotheses=hypotheses)
    (destination / "README.md").write_text(text, encoding="utf-8")
    write_json(destination / "provenance.json", {"run_id": config["run_id"], "settings": {key: config[key] for key in ["question", "written_before_results", "sites", "reference_seed", "nodules", "crop", "ratings", "contact", "split", "activation", "association", "ceiling", "classifier_effect", "clear_feature", "prespecified"]},
                                                 "nodules": {key: nodules[key] for key in ["scans", "scans_with_eligible_nodules", "scans_excluded", "exclusion_reasons", "nodules", "patients", "members", "halves", "readers_per_member"]},
                                                 "sites": summary["sites"], "configuration_sha256": sha256(ROOT / "configs/sae_lidc.json"),
                                                 "source_sha256": {name: sha256(ROOT / "src/sclvmi" / name) for name in ["sae_lidc.py", "sae_context.py", "sae_concepts.py", "sae_interp_candidates.py", "nodule_context.py", "nodule_mask.py"]},
                                                 "exporter_sha256": sha256(Path(__file__)), "python": platform.python_version(), "created_at": timestamp(), "test_used": False})
    print("EXPORTED", destination, flush=True)


if __name__ == "__main__":
    main()

import argparse
import platform
import shutil
from pathlib import Path

import pandas as pd

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


README = """# Nodule SAE features compared with radiomics image features

Run `20261017_sae_radiomics` asks which stable features of the [nodule-interpretation dictionaries](../20261015_sae_nodule_coverage/README.md) (seed 2025; FMCIB layer2 `nodule75`, VISTA3D stage2 `nodule16x`) follow computable image properties of the nodule and its surroundings, which of them change the fixed malignancy classifier, and which combine a confirmed malignancy association, a classifier effect and an incomplete radiomics description. That last group is listed for human review.

![Development R2 of the radiomics explanation](explained_variance.png)

## Procedure

- **Patients:** one nodule record per patient; 1,264 training patients for selection and fitting, 422 development patients for evaluation. The independent test partition is unused.
- **Radiomics:** PyRadiomics on CT resampled to 1 mm, bin width 25 HU.
  - Nodule: shape, first-order and texture features.
  - Rings 0–4 mm and 4–10 mm outside the surface: first-order and texture features.
  - Position and contact: distance and contact fraction to tissue outside the aerated, closed and hole-filled lung (chest wall, mediastinum, diaphragm); the same for large soft-tissue structures (mostly attached vessels); soft-tissue, air and outside-lung fractions of both rings.
  - {features} features, grouped into {groups} groups at training |Spearman| >= 0.9.
- **Targets:** each feature's mean activation within 4 mm of the nodule surface or 4–10 mm outside it, whichever is larger on training patients. The deployed grids are about 7 mm (FMCIB) and 6 mm (VISTA3D) apart, so the nodule interior holds no grid position for many patients.
- **Explanation:** normal scores on training patients, stepwise selection of up to five group representatives (minimum training R2 gain 0.01), R2 on development patients. The shuffled-target control's 99th percentile is about zero.
- **Categories:**
  - Explained: R2 >= 0.5, above the control, and cross-seed agreement >= 0.8.
  - Partly explained: above the control otherwise.
  - Not explained: at or below the control.
  - Features active in fewer than 30 development patients near the nodule are listed separately.
- **Classifier effect:** first-order contribution of each feature to the fixed classifier's logit on development patients (activation times decoder direction dot logit gradient). The 50 largest are removed exactly and the logit recomputed.
- **Review list:** stable, malignancy association beyond nodule volume confirmed in [the candidate run](../20261015_feature_candidates_nodule/README.md), among the 50 largest classifier contributions, and not explained by radiomics.

## Results

{table}

- **Description by radiomics:** most analysable features are partly described, with typical development R2 of 0.1–0.3. Size and surrounding density terms are selected most often.
- **Random-weight control:** random-weight features reach similar R2 (FMCIB median 0.233 against 0.160 for trained features; VISTA3D 0.167 against 0.198). A low R2 therefore does not by itself indicate a concept beyond radiomics, because grid spacing, regional averaging and sparse activation limit the attainable R2.
- **Cross-seed agreement:** nearly every feature exceeds 0.8, so this criterion separates little.
- **Classifier effect:** first-order and exact-removal changes correlate at {agreement}.
  - The classifier effect is concentrated in very few features: FMCIB layer2 3015 (mean logit change on removal −0.40; the next feature −0.06) and VISTA3D stage2 202 (−0.18).
  - Both have confirmed malignancy associations. After nodule volume is controlled, both correlate most with surrounding aerated lung, i.e. lower soft-tissue and outside-lung fractions (Spearman −0.70 and about −0.45).
  - Radiomics describes about half of their variance (R2 0.46 and 0.49).

![Radiomics R2 against classifier contribution](review.png)

## Files

- `categories.csv`: category counts per site and encoder.
- `explanations.csv`: every stable feature with its selected radiomics terms and the terms most correlated before and after controlling nodule volume.
- `trained_features.csv`: the same for trained-encoder features, plus classifier contributions and malignancy confirmation.
- `review_list.csv`: the {review} features for human review.
- `radiomics_groups.csv`: the radiomics feature groups.
- `provenance.json`: run settings and source hashes.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/sae_radiomics.json")
    run = Path(context()[0]["runs"]) / config["run_id"]
    analysis = run / "analysis"
    destination = args.output
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["categories.csv", "explanations.csv", "trained_features.csv", "review_list.csv", "radiomics_groups.csv", "explained_variance.png", "explained_variance.svg", "review.png", "review.svg"]:
        shutil.copyfile(analysis / name, destination / name)
    summary, review = read_json(analysis / "summary.json"), read_json(analysis / "review.json")
    effects = {site: read_json(run / f"effects_{site.replace(':', '_')}.json") for site in config["sites"]}
    categories = pd.read_csv(analysis / "categories.csv")
    medians = summary["median_development_r2"]
    rows = ["| Site | Encoder | Stable | Explained | Partly | Not explained | Active in < 30 patients | Median R2 (analysable) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    explanations = pd.read_csv(analysis / "explanations.csv")
    for row in categories.itertuples():
        analysable = explanations[(explanations.site == row.site) & (explanations.encoder == row.encoder) & (explanations.category != "inactive")]
        rows.append(f"| {row.site} | {row.encoder} | {row.explained + row.partial + row.unexplained + row.inactive:,} | {row.explained} | {row.partial} | {row.unexplained} | {row.inactive:,} | {analysable.development_r2.median():.3f} |")
    agreement = " and ".join(f"{effects[site]['first_order_vs_exact_correlation']:.3f} ({site})" for site in config["sites"])
    review_count = sum(item["review"] for item in review["counts"].values())
    text = README.replace("{features}", str(summary["radiomics_features"])).replace("{groups}", str(summary["radiomics_groups"])).replace("{table}", "\n".join(rows)).replace("{agreement}", agreement).replace("{review}", str(review_count))
    (destination / "README.md").write_text(text, encoding="utf-8")
    write_json(destination / "provenance.json", {"run_id": config["run_id"], "campaign": config["campaign"], "candidate_run": config["candidate_run"], "members": config["members"],
                                                 "settings": {key: config[key] for key in ["spacing_mm", "regions", "region_rule", "pyradiomics", "classes", "contact", "sites", "reference_seed", "activation_statistics", "activation_rule", "analysis", "review", "effects"]},
                                                 "radiomics_features": summary["radiomics_features"], "radiomics_groups": summary["radiomics_groups"], "null_thresholds": summary["null"], "review_counts": review["counts"],
                                                 "effects": {site: {key: value for key, value in item.items() if key in ["dictionary", "patients", "stable_features", "top", "first_order_vs_exact_correlation", "max_probability_difference_torch_vs_classifier"]} for site, item in effects.items()},
                                                 "configuration_sha256": sha256(ROOT / "configs/sae_radiomics.json"), "source_sha256": {name: sha256(ROOT / "src/sclvmi" / name) for name in ["sae_radiomics.py", "nodule_context.py", "nodule_mask.py", "sae_concepts.py", "sae_sites.py"]},
                                                 "exporter_sha256": sha256(Path(__file__)), "python": platform.python_version(), "created_at": timestamp(), "test_used": False})
    print("EXPORTED", destination, flush=True)


if __name__ == "__main__":
    main()

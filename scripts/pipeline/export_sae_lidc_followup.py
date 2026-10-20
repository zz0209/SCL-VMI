import argparse
import platform
import shutil
from pathlib import Path

import pandas as pd

from sclvmi.context import ROOT, context, read_json, sha256, timestamp, write_json


README = """# Follow-up of the LIDC radiologist-rating check: feature pages, finer and coarser sites, and dictionary probes

Run `20261020_sae_lidc_followup` continues [the LIDC check](../20261019_sae_lidc/README.md). Rules were fixed in `configs/sae_lidc_followup.json` before any result; two amendments made before any reader judgement or task item are recorded there.

## Feature pages and reader pass

- **Pages:** for each of the 17 clear features, the 12 highest-activation nodules of the LIDC discovery half and of the LUNA25 training patients, each against a volume-matched low-activation nodule, with the activation map on an axial slice.
- **Pairing rule:** the low partner is the nodule with the lowest activation among the 10 unused nodules nearest in volume. The first rule drew partners from the lowest 20 % of activations, which left the groups unmatched in size.
- **Consistency:** at 30 nodules per group, every confirmed rating of every clear feature differs between high and low nodules in the confirmed direction (`consistency.csv`).
- **Reader pass:** after viewing every page, six features were kept for a blinded reader task.

{reader_table}

The remaining features were set aside for one of these reasons:
- the activation lies on the pleura or chest wall;
- the activation sits at a fixed position of the field;
- the high and low groups show no visible difference;
- the feature repeats a kept pattern.

Several kept features respond on one side of the nodule or in the lung around it, so their maps mark where the feature fires, not the finding.

## Finer and coarser sites

The same rules were applied to the FMCIB layer1 complete-context dictionary (4 mm grid) and the VISTA3D stage3 dictionary (12 mm grid); the classifier-effect criterion is not applied there.

![Largest confirmed, seed-replicated association per rating and site](resolution.png)

{resolution_table}

## Dictionary probes

A lasso on the discovery half predicts each size-adjusted rating from the inputs below; the prediction is correlated with the rating on the confirmation half:
- every stable feature of the dictionary;
- the raw encoder channels;
- the random-weight dictionary;
- the random-weight encoder channels.

![Probe correlations on the confirmation half](probe.png)

{probe_table}

## Files

- `consistency.csv`: high against volume-matched low nodules (30 each) for every confirmed rating, and the LUNA25 malignant share.
- `reader_decisions.csv`: the kept features and the reasons for the others.
- `resolution.csv`: confirmed and seed-replicated associations per site and rating, for trained and random-weight features.
- `associations_extra_sites.csv`: all feature-rating results at the two added sites.
- `probe.csv`: probe correlations with bootstrap intervals and the number of selected inputs.
- `provenance.json`: settings and source hashes.

Pages and reader-task images show individual scans and stay in local storage.
"""


def table(frame, columns, headers, formats):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if index < 2 else "---:" for index in range(len(headers))) + "|"]
    for values in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join("" if pd.isna(value) else format(value, style) for value, style in zip(values, formats)) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/sae_lidc_followup.json")
    run = Path(context()[0]["runs"]) / config["run_id"]
    destination = args.output
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["resolution.csv", "probe.csv", "associations_extra_sites.csv", "resolution.png", "resolution.svg", "probe.png", "probe.svg"]:
        shutil.copyfile(run / "analysis" / name, destination / name)
    shutil.copyfile(run / "pages/consistency.csv", destination / "consistency.csv")
    review = pd.read_csv(ROOT / config["task"]["reader_review"])
    review.to_csv(destination / "reader_decisions.csv", index=False)
    kept = review[review.keep].assign(name=lambda frame: frame.site + " " + frame.feature.astype(str))
    reader_table = table(kept, ["name", "pattern"], ["Feature", "Pattern of the high-activation nodules"], ["", ""])
    resolution = pd.read_csv(run / "analysis/resolution.csv")
    wide = resolution.pivot(index="rating", columns="site", values="trained_replicated_largest").reindex(list(dict.fromkeys(resolution.rating)))
    wide["radiologists"] = resolution.drop_duplicates("rating").set_index("rating").radiologist_agreement
    wide = wide.reset_index()
    sites = ["fmcib:layer1_context", "fmcib:layer2", "vista:stage2", "vista:stage3"]
    resolution_table = table(wide, ["rating", *sites, "radiologists"], ["Rating", "FMCIB layer1 context (4 mm)", "FMCIB layer2 (7 mm)", "VISTA3D stage2 (6 mm)", "VISTA3D stage3 (12 mm)", "One radiologist vs the others"], ["", ".2f", ".2f", ".2f", ".2f", ".2f"])
    probe = pd.read_csv(run / "analysis/probe.csv")
    probe_wide = probe.pivot_table(index=["site", "rating"], columns="input", values="score", sort=False).reset_index()
    probe_table = table(probe_wide, ["site", "rating", "dictionary", "raw", "random_dictionary", "random_raw"], ["Site", "Rating", "Dictionary", "Encoder channels", "Random-weight dictionary", "Random-weight channels"], ["", "", ".2f", ".2f", ".2f", ".2f"])
    text = README.replace("{reader_table}", reader_table).replace("{resolution_table}", resolution_table).replace("{probe_table}", probe_table)
    (destination / "README.md").write_text(text, encoding="utf-8")
    write_json(destination / "provenance.json", {"run_id": config["run_id"], "source_run": config["source_run"], "settings": {key: config[key] for key in ["questions", "written_before_results", "extra_sites", "extra_site_rules", "probe", "pages", "task"]},
                                                 "summary": read_json(run / "analysis/summary.json")["extra_sites"], "configuration_sha256": sha256(ROOT / "configs/sae_lidc_followup.json"),
                                                 "source_sha256": {name: sha256(ROOT / "src/sclvmi" / name) for name in ["sae_lidc_followup.py", "sae_lidc.py", "sae_context.py", "sae_concepts.py", "sae_interp_candidates.py"]},
                                                 "exporter_sha256": sha256(Path(__file__)), "python": platform.python_version(), "created_at": timestamp(), "test_used": False})
    print("EXPORTED", destination, flush=True)


if __name__ == "__main__":
    main()

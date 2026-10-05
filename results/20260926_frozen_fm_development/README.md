# LUNA25 development results · 2026-09-26

- [Selected aggregate results](selected.csv)
- [Candidate aggregate results](ablation.csv)
- [Pipeline and candidate comparison](comparison.png)
- [ROC, precision-recall, and calibration](diagnostics.png)
- [Controlled VISTA3D pooling comparison](pooling_comparison.png)
- [Pooling comparison statistics](pooling_diagnostic_result.json)

Dataset: `FLARE-MedFM/FLARE-AutoMSC`, revision `ce5ea76cde8870805c37614d7ea9b08319b3575e`, `Dataset005_LUNA25`. Training: 3,729 nodule records from 1,264 patients. Development: 1,285 records from 422 patients. Patient groups are disjoint; the independent test partition is reserved.

The selected and candidate tables and first two figures come from local run `20260926_material_preparation/report`. The candidate CSV omits the local machine-path column from its source table. Neural-head AUROC and AP are three-seed means where `seed_count=3`; single-seed/linear entries are identified in the CSV. ROC, precision-recall, and calibration curves use the saved seed-2025 predictors. The pooling figure and JSON come from local run `20260926_frozen_heads/reports/20260926T053546`; they compare mean and max pooling with the same VISTA3D features and head capacity. All figures describe development data. Model choices were made on development data, so these are not independent test estimates.

Files here contain aggregate metrics and visualizations. No record-level predictions, patient identifiers, images, masks, weights, or checkpoints are distributed.

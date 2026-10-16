# SCL-VMI

**Sparse Concept Learning in Volumetric Medical Imaging**

This repository implements lung nodule malignancy classification on the LUNA25 subset of FLARE-AutoMSC. Six pretrained encoders are evaluated with frozen weights. The code exposes their representations for subsequent concept-learning experiments.

## Development results

Each input is centered on the provider's nodule crop. The training partition contains 3,729 records from 1,264 patients. Development uses 1,285 records from 422 patients. Patient groups are disjoint. The independent test partition contains 1,118 records and remains reserved.

| Encoder | Field of view | Classifier | AUROC | AP |
|---|---|---|---:|---:|
| FMCIB | 50 mm | Spatial linear | 0.8787 | 0.4667 |
| CT-FM | 50 mm | Projected max pooling | 0.8506 | 0.4784 |
| VISTA3D | 72 mm | Projected max pooling | 0.8940 | 0.5380 |
| Models Genesis | 32 mm | Projected max pooling | 0.8829 | 0.4494 |
| CoralBay | 32 mm | Adjacent-stage descriptors with linear classification | 0.8606 | 0.4675 |
| TAP-CT | 32 mm | Window CLS and patch summaries with linear classification | 0.8768 | 0.5046 |
| Official LUNA25 I3D, trained locally | 50 mm | Upstream classifier | 0.8938 | 0.5293 |

Development data is used for configuration selection. Neural-head results average three runs with consecutive seeds from 2025 through 2027. Linear heads use deterministic fits. Saved neural predictors use seed 2025. All selected pipelines use a single field of view. TAP-CT processes that field through 12-slice windows.

Direct development assets: [selected metrics](results/20260926_frozen_fm_development/selected.csv), [candidate metrics](results/20260926_frozen_fm_development/ablation.csv), [pipeline comparison](results/20260926_frozen_fm_development/comparison.png), [ROC/PR/calibration](results/20260926_frozen_fm_development/diagnostics.png), and [VISTA3D pooling comparison](results/20260926_frozen_fm_development/pooling_comparison.png). The [result index](results/20260926_frozen_fm_development/README.md) records the evaluation scope.

## Selected SAE dictionaries

Six configuration families were selected through 68 training runs, capacity/activity adjustments, full-development fixed-classifier checks and three-seed diagnostics. Each includes seeds 2025, 2026 and 2027. All 18 selected exports passed reconstruction, activity, classifier and raw-CT checks. Values below show the default seed 2025.

| Encoder / site | Role | Native grid | Features | Target activity | FVU | Probability MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| FMCIB layer1 | Local spatial | 13³ | 2,048 | 128 | 0.07416 | 0.00943 |
| FMCIB layer2 | Regional spatial | 7³ | 4,096 | 256 | 0.09992 | 0.00422 |
| FMCIB global | Whole crop | 1 | 4,096 | 256 | 0.09322 | 0.00475 |
| VISTA3D stage2 | Local spatial | 12³ | 768 | 128 | 0.00885 | 0.00137 |
| VISTA3D stage3 | Regional spatial | 6³ | 1,536 | 256 | 0.01013 | 0.00172 |
| VISTA3D global | Whole crop | 1 | 768 | 384 | 0.08546 | 0.00430 |

The [selection report](results/20261009_sae_selected_materials/README.md) provides the rationale, [three-seed quality figure](results/20261009_sae_selected_materials/selected_quality.png), [all candidates](results/20261009_sae_selected_materials/candidates.csv), [selected metrics](results/20261009_sae_selected_materials/selected.csv), [feature correspondence](results/20261009_sae_selected_materials/seed_consistency.csv) and [PCA references](results/20261009_sae_selected_materials/pca_reference.csv). Development metrics weight patients equally; the independent test set remains reserved.

Spatial BatchTopK dictionaries retain native grids. Global TopK dictionaries return a whole-crop response; their classifier check replaces the final channel mean while preserving the spatial residual. Global PCA has lower reconstruction FVU than the global SAEs. Individual feature correspondence remains limited, particularly for VISTA3D; feature IDs are dictionary-local and medical interpretation needs image evidence. `SpatialSAEPipeline` exposes actual CT activations and geometry. The private selected index is `20261009_sae_asset_selection/selection.json` under configured runs storage.

### Dictionaries for feature interpretation

The [interpretation dictionaries](results/20261014_sae_interpretation_materials/README.md) cover FMCIB layer1 restricted to complete receptive-field positions (k32), FMCIB layer2, VISTA3D stage2 and stage3 (k16), each with eight seeds on the trained encoder and eight on a same-architecture random-weight encoder. All use expansion 4 and 48,000 training steps; training four times longer than the 12,000-step grid raised the number of features that reappear across seeds by 1.9–3.0 times. All 32 trained-encoder dictionaries pass the fixed-classifier checks. Stable features per reference seed are 84, 645, 180 and 325, against 68, 151, 126 and 90 for the random-weight controls. Configuration: `configs/sae_interpretation.json`; [figure](results/20261014_sae_interpretation_materials/selection.png).

With uniformly sampled training tokens, stable features covered the nodule region poorly (FMCIB layer2: −0.185 variance explained at nodule positions against 0.242 beyond 10 mm). [Nodule-region training](results/20261015_sae_nodule_coverage/README.md) draws part of each training batch from positions within 10 mm of training nodules. The dictionaries for nodule interpretation are FMCIB layer2 (expansion 4, k8, three quarters of each batch from the nodule region) and VISTA3D stage2 (expansion 4, k16, half, 192,000 steps); all eight seeds of each pass the fixed-classifier checks. Stable features with nodule enrichment ≥ 3 number 733 and 36, against 49 and 8 for the uniformly trained dictionaries and 341 and 1 for the random-weight controls; stable-feature variance explained at nodule positions is 0.310 and 0.360 (uniform training −0.176 and 0.265). FMCIB layer1 and VISTA3D stage3 keep the uniformly trained dictionaries and are not used for nodule interpretation.

[Feature candidates](results/20261015_feature_candidates/README.md): stable features whose size-adjusted malignancy association, selected on training patients, is confirmed on development patients number 3, 26, 14 and 8 at the four sites, against 3, 0, 0 and 2 for the random-weight controls. Single features add up to 0.043 development AUROC over log nodule volume (0.808) and at most 0.011 over volume plus eight nodule measurements (0.852). The strongest features of FMCIB layer1, layer2 and VISTA3D stage2 and stage3 correlate with one another at 0.61–0.82 across patients.

In the nodule-interpretation dictionaries, [feature candidates](results/20261015_feature_candidates_nodule/README.md) number 36 at FMCIB layer2 and 22 at VISTA3D stage2, against 8 and 0 for the random-weight controls; the random-weight layer2 features add at most 0.007 AUROC over log volume. VISTA3D stage2 feature 202 adds 0.055 [0.031, 0.080] over log volume and 0.024 [0.009, 0.039] over volume plus eight nodule measurements.

An [automatic interpretation pilot](results/20261015_autointerp_pilot/README.md) labelled 24 features with a general vision-language model and scored the labels blind on held-out development patients. Labels for trained-encoder features separate high- from size-matched low-activation patients (median AUROC 0.83–0.90); shuffled controls yield no label. Most labels describe the same coarse contrast, a free solid nodule against an attached one, which random-weight features and measured nodule properties capture as well; the label's own feature ranks first for 3 of 22 labels. VISTA3D stage2 feature 498 (basal lung beside the diaphragm) is the clearest feature-specific label.

### Feature observation materials

The separate `20261008` FMCIB layer1/k64 dictionary has [feature-level research checks](results/20261008_sae_feature_validation/README.md): 128 patient-distinct CT inputs, three SAE seeds, translation and position controls, and patient-balanced correspondence. Thirteen features pass the numerical screens; nine provide repeated broad image-content descriptions as research starting points. The [feature catalog](results/20261008_sae_feature_validation/reviewed_features.csv) retains each description and its qualifications. These are conditional development-set observations from one AI image reviewer; clinical identity and feature-specific causal effects remain untested. The private `20261008_feature_validation/fmcib/research_materials.json` contains fixed weights and CT evidence paths. Its [dictionary study](results/20261008_spatial_sae/README.md) records the associated earlier training results.

## Engineering entry points

The [dataset and SAE heatmap viewers](local_tools/viewers/viewer.md) provide
English/Chinese interfaces for local FLARE-AutoMSC images and a shared heatmap
browser with configuration, seed, Feature and image-scope selectors. Viewer source
and launch instructions are included;
datasets, dictionaries and private analysis assets are supplied through local
storage configuration.

| Component | Source |
|---|---|
| Storage initialization | [initialize_storage.py](scripts/data/initialize_storage.py) |
| Dataset acquisition | [download_flare.py](scripts/data/download_flare.py) |
| Environment setup | [bootstrap_sources.ps1](scripts/pipeline/bootstrap_sources.ps1), [setup_environment.ps1](scripts/pipeline/setup_environment.ps1), [finish_environment.ps1](scripts/pipeline/finish_environment.ps1) |
| Additional model dependencies | [prepare_frozen_heads.ps1](scripts/pipeline/prepare_frozen_heads.ps1) |
| Frozen-head comparison | [fit_frozen_heads.py](scripts/pipeline/fit_frozen_heads.py) |
| Local-field comparison | [prepare_materials.py](scripts/pipeline/prepare_materials.py) |
| Image and feature inference | [material_predict.py](src/sclvmi/material_predict.py) |
| Spatial SAE training and inference | [sae.py](src/sclvmi/sae.py), [sae_predict.py](src/sclvmi/sae_predict.py) |
| Multi-layer SAE comparison | [Input preparation](src/sclvmi/sae_campaign_data.py), [resumable training](src/sclvmi/sae_campaign_train.py), [fixed-classifier evaluation](src/sclvmi/sae_campaign_evaluate.py), [candidate selection](src/sclvmi/sae_campaign_compare.py), [PCA and seed diagnostics](src/sclvmi/sae_campaign_diagnostics.py) |
| Interpretation dictionaries | [inputs and random-weight encoders](src/sclvmi/sae_context.py), [cross-seed stability](src/sclvmi/sae_interp_stability.py), [fixed-classifier checks](src/sclvmi/sae_interp_evaluate.py), [feature statistics](src/sclvmi/sae_interp_profiles.py), [selection](src/sclvmi/sae_interp_select.py), [nodule-region pools](src/sclvmi/sae_nodule_pool.py) and [coverage](src/sclvmi/sae_interp_coverage.py), [feature candidates](src/sclvmi/sae_interp_candidates.py) and [their follow-up](src/sclvmi/sae_interp_followup.py), [automatic interpretation](src/sclvmi/sae_autointerp.py) with [model calls and validation](src/sclvmi/sae_autointerp_model.py), [whole-CT maps](local_tools/viewers/feature_explorer/interpretation_whole.py) |
| Official baseline adapters | [luna25.py](src/sclvmi/luna25.py), [automsc.py](src/sclvmi/automsc.py) |
| Dataset and heatmap viewers | [Viewer setup and local asset requirements](local_tools/viewers/viewer.md) |

The tested environment uses Python 3.12 with PyTorch 2.8.0 and CUDA 12.8 on Windows. Dependencies are specified in [pipeline.json](configs/environments/pipeline.json). Storage initialization creates the local path configuration. Dataset acquisition requires an approved Hugging Face account. Model revisions and license information are recorded in [model_sources.json](configs/model_sources.json).

Sampling settings are defined in [pipeline.json](configs/pipeline.json). The local-field experiment uses [material_preparation.json](configs/material_preparation.json). Per-record caches support interrupted extraction. Neural heads save resumable checkpoints.

## Feature interface

```python
from sclvmi.material_predict import MaterialPipeline

pipeline = MaterialPipeline(run_directory, kind)
features = pipeline.extract(image_path)
probability = pipeline.predict_features(features)
```

`kind` matches the saved run type (`reference` / `descriptor` / `local_max`). `training_features()` iterates over training records. Feature bundles retain their spatial or window layout. `predict_features()` accepts values in the encoder's original units.

All six selected pipelines passed raw-image inference checks on two benign and two malignant records per model. Cached-feature predictions were checked against the same saved results. The maximum absolute probability difference was 1.10×10⁻⁷.

Data and model assets are stored outside the repository. Their use remains subject to the original providers' terms.

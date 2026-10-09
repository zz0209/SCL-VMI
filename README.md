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

## Spatial SAE dictionaries

BatchTopK dictionaries preserve native spatial grids for CT crop heatmaps. Each configuration passed reconstruction, sparsity and complete-development classifier checks for seeds 2025, 2026 and 2027. The table shows the default seed-2025 exports, evaluated with the same saved classifier across SAE seeds.

| Encoder / site | Native grid | Dictionary size | Target mean activity | FVU | Mean cosine | Probability MAE |
|---|---:|---:|---:|---:|---:|---:|
| VISTA3D / stage2 | 12³ | 768 | 80 | 0.04295 | 0.95580 | 0.00293 |
| FMCIB / layer1 | 13³ | 2,048 | 64 | 0.11884 | 0.96179 | 0.00888 |

Raw-CT inference checks passed for all six exports, including activation maps, physical coordinates, saved predictions and CPU/GPU reconstruction. Whole-dictionary reconstruction is consistent across seeds. Individual feature correspondence is limited: full-vector matching gives median activation correlations of 0.22–0.23 for VISTA3D and 0.33–0.34 for FMCIB. Feature IDs remain dictionary-local; anatomical interpretation requires image evidence.

The [SAE results and design choices](results/20261008_spatial_sae/README.md) include [all selected metrics](results/20261008_spatial_sae/selected.csv), [candidate comparisons](results/20261008_spatial_sae/candidate_comparison.csv), [training curves](results/20261008_spatial_sae/training_quality.png), [capacity/activity comparisons](results/20261008_spatial_sae/candidate_quality.png), and [PCA/seed diagnostics](results/20261008_spatial_sae/diagnostics.json). `SpatialSAEPipeline` returns activation arrays and native-grid RAS coordinates. The private dictionary index is stored under the configured runs directory at `20261008_spatial_sae/dictionary_index.json`.

The FMCIB default dictionary also has [feature-level research checks](results/20261008_sae_feature_validation/README.md): 128 patient-distinct CT inputs, three SAE seeds, translation and position controls, and patient-balanced correspondence. Thirteen features pass the numerical screens; nine provide repeated broad image-content descriptions as research starting points. The [feature catalog](results/20261008_sae_feature_validation/reviewed_features.csv) retains each description and its qualifications. These are conditional development-set observations from one AI image reviewer; clinical identity and feature-specific causal effects remain untested. The private `20261008_feature_validation/fmcib/research_materials.json` contains fixed weights and CT evidence paths.

## Engineering entry points

The [dataset and SAE heatmap viewers](local_tools/viewers/viewer.md) provide
English/Chinese interfaces for local FLARE-AutoMSC images, nodule-crop responses
and whole-CT responses. Viewer source and launch instructions are included;
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

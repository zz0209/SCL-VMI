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

Aggregate result data and direct visualizations are available in [development results](results/20260926_frozen_fm_development/README.md).

## Engineering entry points

| Component | Source |
|---|---|
| Storage initialization | [initialize_storage.py](scripts/data/initialize_storage.py) |
| Dataset acquisition | [download_flare.py](scripts/data/download_flare.py) |
| Environment setup | [bootstrap_sources.ps1](scripts/pipeline/bootstrap_sources.ps1), [setup_environment.ps1](scripts/pipeline/setup_environment.ps1), [finish_environment.ps1](scripts/pipeline/finish_environment.ps1) |
| Additional model dependencies | [prepare_frozen_heads.ps1](scripts/pipeline/prepare_frozen_heads.ps1) |
| Frozen-head comparison | [fit_frozen_heads.py](scripts/pipeline/fit_frozen_heads.py) |
| Local-field comparison | [prepare_materials.py](scripts/pipeline/prepare_materials.py) |
| Image and feature inference | [material_predict.py](src/sclvmi/material_predict.py) |
| Official baseline adapters | [luna25.py](src/sclvmi/luna25.py), [automsc.py](src/sclvmi/automsc.py) |

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

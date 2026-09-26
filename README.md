# SCL-VMI

**Sparse Concept Learning in Volumetric Medical Imaging**

SCL-VMI explores sparse concept representations in volumetric medical imaging, with an initial focus on the [FLARE-AutoMSC dataset](https://huggingface.co/datasets/FLARE-MedFM/FLARE-AutoMSC). The research scope includes CT, multi-sequence MRI, and PET/CT.

The initial task is lung nodule malignancy classification on the LUNA25 subset. This repository provides dataset acquisition, patient-grouped splits, upstream baseline adapters, frozen volumetric encoders, feature caching, classification heads, and evaluation. SAE methods remain under investigation.

## Implemented pipelines

| Pipeline | Implementation and validation |
|---|---|
| AutoMSC | Pinned upstream nnU-Net joint segmentation/classification trainer and inference; real-data training, checkpoint reload, and segmentation/classification inference verified. |
| LUNA25 I3D | Pinned upstream conversion, sampling, model, and training settings; real-data batch-32 training, checkpoint reload, and single-image inference verified. |
| Frozen encoders | FMCIB, CT-FM, VISTA3D/NV-Segment-CT, Models Genesis, CoralBay, and TAP-CT-B-3D with published checkpoints, spatial features, and linear/MLP/pooling/attention classifiers. |
| Concept bottleneck | Independent Ridge + additive EBM implementation of the approach described in arXiv:2608.07857v1. Exact reproduction is pending author code, cohort mapping, concept labels, and nodule-size provenance. Real concept training has not been validated. |

The upstream adapters use the FLARE cohort and project patient split. They are method reproductions with documented runtime adaptations; they do not reproduce published numerical results. The AutoMSC published validation archive contains Glioma checkpoints, so its LUNA25 model must be trained. No trained model or medical record is redistributed here.

## Getting started

Use Python 3.10 or newer in a dedicated environment:

```bash
python -m pip install -r requirements-data.txt
```

Create storage outside this repository. The following is a Windows example; on other systems, supply an absolute path to a suitable mounted volume.

```bash
python scripts/data/initialize_storage.py --storage-root D:/SCL-VMI-storage
python scripts/data/probe_hub.py
```

The initializer creates `configs/storage.local.json` and directories for source data, derived data, activations, checkpoints, and run outputs. Local configuration and generated files are excluded from version control. Use `configs/storage.example.json` to inspect the configuration structure.

## Dataset access and download

Access must be approved by the dataset provider. Log in with an approved Hugging Face account before downloading. Authentication and downloads must use the same `HF_HOME` value from `configs/storage.local.json`.

PowerShell example:

```powershell
$storage = Get-Content -Raw configs/storage.local.json | ConvertFrom-Json
$env:HF_HOME = $storage.hf_home
hf auth login
python scripts/data/download_flare.py --preflight-only
python scripts/data/download_flare.py --smoke-test
python scripts/data/download_flare.py
```

On other shells, set `HF_HOME` to the same configured path before running `hf auth login` and the Python commands.

The download configuration pins repository revision `ce5ea76cde8870805c37614d7ea9b08319b3575e`: 29,059 files totaling 81,744,497,845 bytes (approximately 76.13 GiB), including the published validation directory. Allocate at least 100 GiB for acquisition; derived representations require additional space. The download does not include unpublished test data.

The smoke test downloads representative image channels, labels, and metadata across the dataset folders and verifies their content hashes. Its files are reused by the full download.

The full downloader obtains authenticated file hashes, downloads the complete snapshot without subset filters, and checks every file against its expected size and digest. Re-running the command resumes interrupted transfers. File progress is displayed during transfer, and each verification pass saves a separate receipt. Only a fully verified snapshot receives a `complete` status. Verification errors are reported without automatically deleting files.

After the authenticated inventory has been saved, an offline verification pass is available:

```bash
python scripts/data/download_flare.py --verify-only
```

Acquiring all subsets does not define a training split. Preserve the provider's validation role and establish patient-level grouping, longitudinal grouping, and modality pairing before model development. See the dataset provider for access conditions and subset licenses. No dataset files are redistributed by this repository.

## Download tests

```bash
python scripts/data/test_download_integrity.py
```

These offline tests cover content hashes, redacted metadata, manifest mismatches, and path containment. They do not replace full-snapshot verification.

## Pipeline environment

The tested runtime is Windows, Python 3.12, PyTorch 2.8.0/CUDA 12.8, and MONAI 1.5.1. A CUDA GPU is required for the model commands. Initialize storage and acquire the dataset before running them. From the repository root:

```powershell
./scripts/pipeline/bootstrap_sources.ps1
./scripts/pipeline/setup_environment.ps1 -BasePython C:/path/to/python312/python.exe
./scripts/pipeline/finish_environment.ps1
$storage = Get-Content -Raw configs/storage.local.json | ConvertFrom-Json
$python = Join-Path $storage.environments 'pipeline/Scripts/python.exe'
& $python scripts/pipeline/model_assets.py download
& $python -m sclvmi manifest
& $python -m pytest tests/test_pipeline_data.py -q
& $python scripts/pipeline/validate_fms.py --run-id fm_validation_20260925_s2025
```

Dependencies are specified in `configs/environments/pipeline.json`; source commits are pinned in the bootstrap script and model releases in `configs/model_sources.json`. Downloads include roughly 7 GB of published model assets. Additional free space is needed for environments, converted images, and checkpoints. Check local GPU, memory, and disk availability before training.

Weights retain their original licenses: FMCIB CC BY 4.0, CT-FM Apache 2.0, NV-Segment-CT NVIDIA model terms, and Models Genesis ASU non-commercial terms. Consult each upstream release before use or redistribution.

## Sampling and data split

The fixed snapshot yields 6,132 available LUNA25 records. Clinical rows without an image are excluded explicitly. Provider fold 0 is held out for testing (1,118 records); fold 1 is development (1,285); folds 2–4 are training (3,729). Patient, nodule, and scan groups are disjoint across these splits. Repeated observations remain within their patient group.

Each classification input uses the physical center of the provider's nodule crop. Segmentation masks are not used to define FM inputs. Resampling preserves physical coordinates; padding supplies missing field of view. Encoder spacing and normalization follow their releases where specified. CT-FM and VISTA classification crop sizes, and the Genesis physical field of view, are project choices based on available nodule-image coverage.

These splits control downstream training and selection. Overlap with foundation-model pretraining cohorts has not been ruled out.

| Encoder | Input shape | Spacing (mm) | Orientation | Embedding |
|---|---:|---:|---|---:|
| FMCIB | 50 × 50 × 50 | 1 × 1 × 1 | upstream LPS transform | 4,096 |
| CT-FM | 24 × 64 × 64 | 3 × 1 × 1 | SPL | 512 |
| VISTA3D | 48 × 48 × 48 | 1.5 × 1.5 × 1.5 | RAS | 768 |
| Models Genesis | 64 × 64 × 32 | 0.78125 × 0.78125 × 1.5625 | RAS | 512 |

`configs/pipeline.json` specifies HU normalization and all head choices. Feature extraction uses batch size one and float32, with bounded two-case preloading for MONAI inputs; the upstream FMCIB ITK transform runs serially. Global mean pooling produces a vector per record; `--spatial` also retains the encoder feature grid. Spatial grids have model-dependent resolution and are not segmentation predictions.

## Frozen encoder training and inference

```powershell
& $python scripts/pipeline/fit_frozen_baselines.py --run-id frozen_20260925_s2025
```

This extracts all training/development embeddings and trains all four heads. Per-record caches resume after interruption, and completed heads are reused. If interrupted during head fitting, preserve that incomplete run and use a fresh run ID; cached features remain reusable. The scaler is fitted on training data. Six regularization values are compared using development AUROC, with log loss as the tie-breaker. Development scores therefore include model selection and are not final test estimates.

Individual commands are also available:

```powershell
& $python -m sclvmi extract --model fmcib --splits train development --spatial
& $python -m sclvmi head --features <feature-directory> --run-id <unique-run-id>
& $python -m sclvmi.predict fm --model fmcib --head <head-run-directory> --image <nodule-image.nii.gz>
& $python -m sclvmi.evaluation --predictions <development_predictions.csv> --output <new-evaluation.json>
```

Evaluation reports AUROC, average precision, Brier score, and log loss, with patient-cluster bootstrap intervals for AUROC. All outputs remain in configured external storage. Use a new run ID when changing data or settings.

## Six-encoder frozen-head comparison

The frozen-head experiment adds CoralBay and TAP-CT-B-3D to FMCIB, CT-FM, VISTA3D, and Models Genesis. Install the pinned EVA vision environment and acquire the additional weights:

```powershell
./scripts/pipeline/prepare_frozen_heads.ps1
& $python scripts/pipeline/fit_frozen_heads.py --models coralbay tapct --smoke-only
& $python scripts/pipeline/fit_frozen_heads.py
```

The experiment compares standardized native embeddings with logistic regression and an MLP, plus spatial mean, max, gated-attention, and spatial linear heads. Spatial linear preserves a 2³ pooled grid; TAP-CT preserves its ordered window vectors. The attention implementation is imported from the pinned EVA package. Encoders remain frozen. Neural heads use two learning rates and repeat the selected configuration with three seeds. Selection uses development AUROC, with log loss as a tie-breaker. Checkpoints and per-case feature caches support interruption recovery.

CoralBay uses a 64³ crop at 1.5 mm, HU clipping to [-1000, 1000], and the release's multiscale embedding; spatial heads consume its last-stage grid. TAP-CT uses a 50 mm LPS nodule crop at 1 mm, its official intensity processor, and 12-slice axial windows at 224². Window CLS vectors form the classification bag. These sampling settings are project configurations; pretrained releases and source revisions are pinned in `configs/model_sources.json`.

A matched CoralBay field-of-view comparison uses `--models coralbay --field-mm 50 --variant coralbay_fov50`, retaining 64³ inputs and the same normalization and heads.

After these runs complete, `scripts/pipeline/fit_balanced_heads.py` compares class-balanced BCE using each encoder's selected neural head and learning rate. The positive weight comes from the training class ratio. Three seeds are evaluated without an additional parameter search.

`scripts/pipeline/fit_tree_head.py --model fmcib` evaluates one fixed shallow histogram-gradient-boosting classifier on cached FMCIB embeddings, including a real-data smoke check and checkpoint reload validation.

```powershell
& $python scripts/pipeline/validate_upstream_heads.py --task fmcib
& $python scripts/pipeline/validate_upstream_heads.py --task fmcib-inference
& $python scripts/pipeline/validate_upstream_heads.py --task eva
& $python -m sclvmi.frozen_predict --run <head-run-directory> --image <nodule-image.nii.gz>
& $python -m sclvmi.frozen_predict --run <head-run-directory> --verify
```

The FMCIB command runs the upstream logistic-regression objective on the project's existing training/development split. The EVA command exercises its native training and checkpoint APIs on real cached CoralBay features. Image verification recomputes predictions from NIfTI inputs and compares them with saved development predictions. CoralBay weights retain their MIT license; TAP-CT weights retain CC BY-NC 4.0 and its model source Apache 2.0.

All six selected pipelines and the official FMCIB linear adaptation passed raw-image inference checks using two benign and two malignant records each. Fitted heads also passed reload checks on their development predictions.

## VISTA3D classification result

The frozen VISTA3D image encoder with projected spatial max pooling reaches **0.8940 mean development AUROC** and **0.5380 mean average precision** across seeds 2025, 2026, and 2027. The locally trained official I3D reference reaches 0.8938 AUROC and 0.5293 average precision on the same development records. Configuration selection uses the development fold; the independent test fold is reserved.

The input is a nodule-centered 72 mm field sampled at 1.5 mm into 48³ voxels. The encoder produces 27 spatial vectors with 768 channels. Training-derived normalization, a shared 768→128 projection, channelwise spatial maximum, and a 128→64→1 classifier give 106,753 trainable parameters. The encoder weights remain frozen.

| Seed | AUROC | Average precision |
|---|---:|---:|
| 2025 | 0.8934 | 0.5210 |
| 2026 | 0.8951 | 0.5357 |
| 2027 | 0.8935 | 0.5572 |

The same-input, same-parameter-count mean-pooling control reaches AUROC 0.8343 with seed 2025. The max-minus-mean AUROC difference is 0.0590, with a paired patient-bootstrap 95% interval of [0.0329, 0.0892] on the development fold. The experiment retains per-record predictions and all seed checkpoints in configured run storage.

```powershell
$run = Join-Path $storage.runs '20260926_frozen_heads/vista_max_c1_s2025'
& $python -m sclvmi.frozen_predict --run $run --image <nodule-image.nii.gz>
& $python -m sclvmi.frozen_predict --run $run --verify
```

## Common encoder adaptation

All four encoders support the same three conventional comparisons in `configs/adaptation.json`:

| Arm | Trainable components | Training augmentation |
|---|---|---|
| `frozen_mlp` | Standardized embedding → 256 → 1, GELU, dropout 0.1 | None |
| `last_block` | Final encoder block and linear classifier | None |
| `last_block_translation` | Final encoder block and linear classifier | Uniform translation up to 2.5 mm per input axis |

Each arm retains the encoder's existing preprocessing and global mean pooling. Inference uses one center crop. Translation interpolates the prepared crop with zero padding. Fine-tuning initializes the classifier from its frozen logistic baseline and uses AdamW, an effective batch of 32, and ten epochs. Models Genesis uses a microbatch of one because its upstream normalization uses current-batch statistics during evaluation; gradient accumulation preserves the effective batch size. Other encoders use microbatches of eight. Frozen MLP image inference retains the cuDNN TF32 setting used to create the original embeddings; fine-tuning disables TF32.

```powershell
& $python scripts/pipeline/fit_adaptation.py --model fmcib --arm frozen_mlp --candidate 0 --seed 2025 --baseline <frozen-head-run-id> --run-id <unique-adaptation-run-id>
& $python -m sclvmi.adaptation_predict --run-id <adaptation-run-id> --image <nodule-image.nii.gz>
```

MLP candidates enumerate three learning rates × two weight decays (indices 0–5). Each fine-tuning arm has two encoder learning rates (indices 0–1). Select candidates by development AUROC with log loss as tie-breaker, then repeat the selected configuration with seeds 2026 and 2027. Epoch checkpoints, optimizer state, random state, predictions, and progress are saved in external run storage; reusing the same run ID resumes its unchanged configuration. Completed checkpoints are reloaded and development predictions are recomputed. Real-data training and independent raw-image inference checks pass for all four encoders. This adaptation grid is paused; the current comparison uses frozen encoders and the head experiments described above.

## Upstream supervised baselines

Functional checks on real data:

```powershell
& $python -m sclvmi.luna25 --run-id i3d_smoke_20260925_s2025 --limit 8 --epochs 1 --batch-size 2
& $python -m sclvmi.automsc prepare --run-id automsc_smoke_20260925_s2025 --limit 8
& $python -m sclvmi.automsc train --run-id automsc_smoke_20260925_s2025 --epochs 1 --steps 2
& $python -m sclvmi.predict automsc --source-run automsc_smoke_20260925_s2025 --output-id automsc_predictions_20260925_s2025 --limit 2
& $python -m sclvmi.predict i3d --checkpoint <i3d-run-directory>/best.pt --image <nodule-image.nii.gz>
```

For full training, omit the sample limits and smoke iteration overrides, using a fresh run ID. I3D defaults to the upstream ten epochs and batch size 32; this schedule has completed locally. AutoMSC uses its upstream training schedule and a memory-aware nnU-Net plan; its fingerprint is fitted on training images only. Both save epoch checkpoints and accept `--resume` with the original configuration. The AutoMSC long training run remains paused.

I3D can also resume preparation before the first checkpoint. AutoMSC development inference accepts `--all-development --resume`; completed case receipts are reused only for the same checkpoint and case selection.

The I3D adapter retains 50 mm / 64³ sampling, HU clipping, augmentation, balanced sampling, Adam, and frozen batch normalization. Runtime changes support the current GPU, serial loading, deterministic development ordering, and resumable checkpoints. I3D uses float32 with TF32 and cuDNN benchmarking disabled for consistent per-case inference. AutoMSC uses upstream architecture, loss, augmentation, trainer, and sliding-window inference with project data staging and resource settings.

## Concept bottleneck inputs

`python -m sclvmi.cbm --help` lists the independent implementation's required inputs. It requires verified LIDC reader concepts, their feature files, and a patient-grouped LUNA manifest containing a documented `size_mm` measurement. The FLARE masks do not supply those eight reader concepts. Results from this implementation must be identified as an independent reconstruction until the paper's exact assets and unspecified choices can be verified.

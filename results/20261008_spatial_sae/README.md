# Spatial SAE dictionaries for VISTA3D and FMCIB

This experiment prepares spatial sparse autoencoders for subsequent CT crop heatmaps. It measures reconstruction of frozen encoder features, preservation of the existing malignancy classifier, and reproducibility across SAE initializations.

## Measured results

All six dictionaries passed the recorded reconstruction, sparsity and complete-development classifier criteria. The table shows the predetermined seed-2025 exports. Full three-seed values are in [selected.csv](selected.csv).

| Encoder / site | Dictionary / target activity | FVU | Mean cosine | Probability MAE | Original → reconstructed AUROC | Original → reconstructed AP |
|---|---:|---:|---:|---:|---:|---:|
| VISTA3D / stage2 | 768 / 80 | 0.042949 | 0.955798 | 0.002926 | 0.893352 → 0.892227 | 0.521002 → 0.522733 |
| FMCIB / layer1 | 2,048 / 64 | 0.118843 | 0.961794 | 0.008881 | 0.878673 → 0.881633 | 0.466736 → 0.468374 |

Default probability-MAE 95% paired-patient bootstrap intervals are [0.002611, 0.003260] for VISTA3D and [0.007942, 0.009874] for FMCIB. Across all three seeds, VISTA3D FVU is 0.042892–0.042949 and FMCIB FVU is 0.118541–0.119290. AUROC/AP change intervals include zero for the default exports.

All six exports passed raw-CT activation-map, geometry and prediction comparisons, CPU/GPU reconstruction checks, checkpoint reload and inference-batch checks. Training continuation was tested against uninterrupted training on real vectors and produced bit-identical parameters. Independent physical-coordinate checks covered eight training images per encoder.

### Feature repeatability

The complete 108,032-vector development comparison gives the following ranges over the three seed pairs:

| Encoder | Median matched activation correlation | Matched features with correlation ≥ 0.8 | Default features active at fewer than 0.1% of sampled positions |
|---|---:|---:|---:|
| VISTA3D | 0.2234–0.2297 | 2.86–3.78% | 278 / 768 |
| FMCIB | 0.3299–0.3353 | 4.30–5.03% | 325 / 2,048 |

Whole-dictionary reconstruction is repeatable; individual feature correspondence is limited. These exports support exploration of crop heatmaps with dictionary-local feature IDs. Anatomical labels and claims about stable individual concepts require direct image evidence and confirmation across dictionaries. Rare features require sufficient image examples for interpretation.

## Data and evaluation

The established LUNA25 partitions contain 3,729 training records from 1,264 patients and 1,285 development records from 422 patients. Training and development patient groups are disjoint. The independent test partition remains reserved. Development data supports configuration selection; reported metrics and bootstrap intervals describe that development population.

Each patient contributes 512 training or 256 development vectors, giving 647,168 training and 108,032 development vectors at each candidate site. Records and spatial positions are sampled within patient. Classifier substitution uses every location in every development record.

All SAE seeds use the same saved classifier for each encoder. VISTA3D uses its saved seed-2025 classifier; FMCIB uses its deterministic spatial linear classifier.

## Design choices

| Choice | Reason |
|---|---|
| Frozen VISTA3D and FMCIB with their saved classifiers | Direct substitution measures how much the dictionary preserves the existing prediction path. |
| VISTA3D stage2, with 192 channels on a 12³ grid | The matched stage2/stage3 pilot favors stage2 in reconstruction and classifier preservation, while retaining more spatial locations. |
| FMCIB layer1, with 512 channels on a 13³ grid | It provides better reconstruction and denser spatial sampling than layer2 in the matched pilot. The selected full-budget configuration also satisfies classifier checks. |
| One shared dictionary applied to each spatial channel vector | Every activation retains its native location and a consistent feature definition throughout a crop. |
| BatchTopK, dictionary expansion 4 | Average activity can be controlled directly while allowing spatial locations to use different numbers of features. |
| Target mean activity 80 for VISTA3D and 64 for FMCIB | These configurations meet the recorded reconstruction and coverage targets within the tested capacity/activity range. |
| Equal sampling weight per patient | Patients with more records do not dominate the dictionary fit. |
| Training channel mean and one global RMS scale | Centering aids optimization while retaining relative channel variation and conversion to original feature units. |
| A fixed inference threshold calibrated on training vectors | Heatmaps do not depend on which other images or locations share the inference batch. |
| 12,000 steps and seeds 2025, 2026 and 2027 | Repeated training measures optimization consistency and feature correspondence. Seed 2025 is the predetermined default export. |
| Original model preprocessing and physical RAS coordinates | Crop position, orientation, padding and network stride determine correspondence to source CT. |

The training configuration uses Adam with learning rate 0.0003, batch size 2,048, 200 warmup steps and decay during the final 20% of training. Dictionaries use float32 arithmetic with TF32 disabled. [sae_preparation.json](../../configs/sae_preparation.json) records the complete base configuration; each run request records its actual activity and capacity.

Before observing trained dictionaries, the preparation criteria were FVU ≤ 0.15, mean raw-token cosine ≥ 0.95, inactive development-feature fraction ≤ 2%, mean activation count within 20% of target, probability MAE ≤ 0.025, AUROC decrease ≤ 0.01 and AP decrease ≤ 0.02. These are project preparation criteria. Anatomical meaning requires subsequent image and annotation evidence.

## Result files

- `selected.csv`: reconstruction, sparsity, full classifier metrics and 1,000-resample paired-patient bootstrap intervals for each selected dictionary.
- `candidate_comparison.csv`: all completed scientific candidates and pilots, with training budget and explicit classifier-evaluation scope. Blank classifier metrics mean that evaluation was not performed.
- `training_quality.png` / `.svg`: selected-run reconstruction and activity trajectories.
- `candidate_quality.png` / `.svg`: seed-2025 capacity/activity comparisons at a matched 12,000-step budget.
- `diagnostics.json`: train-fitted rank-k PCA references, the initial 8,440-vector comparison, and the complete 108,032-vector cross-seed comparison with activation-frequency counts.

PCA uses the same training vectors and k scalar coefficients. The SAE uses a larger dictionary and variable sparse support; the two representation families have different storage requirements. Cross-seed comparisons use maximum-total activation-correlation matching. Feature identifiers remain dictionary-local.

Rank-80 PCA for VISTA3D gives FVU 0.202501 and cosine 0.884070; rank-64 PCA for FMCIB gives FVU 0.280201 and cosine 0.918327. These are reconstruction references under the stated coefficient-count comparison.

## Inference and reproduction

```python
from sclvmi.sae_predict import SpatialSAEPipeline

pipeline = SpatialSAEPipeline(dictionary_path)
bundle = pipeline.extract(image_path)
activations = bundle["activations"]
geometry = bundle["geometry"]
```

`activations` has shape `(dictionary_size, grid_x, grid_y, grid_z)`. `geometry["grid_affine_ras"]` maps native-grid indices to physical RAS coordinates. The bundle also includes the preprocessed input crop and original/reconstructed classifier probabilities. A grid location's receptive field includes surrounding image content.

The configured run directory contains `20261008_spatial_sae/dictionary_index.json`, with the default and repeated-seed dictionary paths. Model weights, source CT, per-record activations and predictions remain in private storage.

Use the configured pipeline environment after [source setup](../../scripts/pipeline/bootstrap_sources.ps1). The execution entry points are:

| Operation | Entry point |
|---|---|
| Spatial feature extraction and patient sampling | `python -m sclvmi.sae_data --model vista` or `--model fmcib` |
| Dictionary training | `python -m sclvmi.sae --model MODEL --site SITE --run-id RUN_ID --steps 12000 --k K --expansion 4 --seed SEED` |
| Complete classifier substitution | `python -m sclvmi.sae_evaluation --run-id RUN_ID` |
| Patient bootstrap, PCA and seed comparison | `python -m sclvmi.sae_diagnostics --runs RUN_2025 RUN_2026 RUN_2027 --name DIAGNOSTIC_NAME` |
| Complete-vector seed comparison | `python -m sclvmi.sae_diagnostics --runs RUN_2025 RUN_2026 RUN_2027 --name COMPARISON_NAME --comparison-only --tokens-per-patient 256` |
| Raw-CT inference verification | `python scripts/pipeline/check_spatial_sae.py --runs RUN_IDS` |
| Saved-result figures and aggregate tables | `python scripts/pipeline/summarize_spatial_sae.py --runs RUN_IDS --diagnostics DIAGNOSTIC_NAMES --full-comparisons COMPARISON_NAMES --output OUTPUT_DIRECTORY` |

Actual run identifiers and hyperparameters are recorded in `selected.csv`. New experiments require distinct run IDs. The figure script reads saved results and does not refit dictionaries or choose configurations.

## Implementation sources

- [BatchTopK Sparse Autoencoders](https://arxiv.org/html/2412.06410v1), Bussmann, Leask and Nanda (2024): [author implementation](https://github.com/bartbussmann/BatchTopK), pinned to commit `b9aab1c6156381ae7ae2997e3490e7b99e195dde`, MIT license.
- [PatchSAE](https://github.com/dynamical-inference/patchsae), Lim, Choi, Choo and Schneider, ICLR 2025: spatial decomposition and reconstruction/classifier checks inform this preparation.
- [Dora-SAE](https://feature3d.github.io/Dora-SAE/), Miao, Zhou, Zhou and Oztireli, ICLR 2026: spatially indexed activations and repeated-dictionary comparisons inform the design.

The preparation reuses the pinned BatchTopK module. The medical encoder pipelines retain their original model and data usage terms.

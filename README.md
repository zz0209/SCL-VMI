# SCL-VMI

**Sparse Concept Learning in Volumetric Medical Imaging**

This repository implements lung nodule malignancy classification on the LUNA25 subset of FLARE-AutoMSC. Ten pretrained encoders are evaluated with frozen weights, alongside the trained official LUNA25 I3D and AutoMSC baselines. The code exposes their representations for subsequent concept-learning experiments.

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
| FlexiCT-3D | 32 mm | Projected max pooling on block 10 of 16 | 0.9191 | 0.5727 |
| FlexiCT-3D-VLM image encoder | 40 mm | Projected max pooling on block 10 of 16 | 0.9129 | 0.5437 |
| nnFoundation ViT | 32 mm | Projected max pooling | 0.8897 | 0.5263 |
| nnFoundation CNN | native crop | Three-stage mean+max with linear classification | 0.7656 | 0.2636 |
| AutoMSC | Provider crop with official preprocessing | Official joint classification and segmentation model | 0.8797 | 0.5108 |

The nnFoundation and FlexiCT rows come from the [encoder comparison](results/20261017_encoder_pipeline/README.md).
- **Procedure:** the same reference-field, local-field, descriptor and head procedure as for the existing encoders.
- **FlexiCT:** an added comparison of fields (32–64 mm) and layers (blocks 4–16) found block 10 with 32–40 mm fields best. With the final block and the standard procedure, FlexiCT gave 0.8966 and 0.8945. The FlexiCT rows involve more development-based selection than the others.
- **nnFoundation CNN:** the masked-autoencoding CNN stays far lower with every frozen input and head tried; its paper evaluates classification with full fine-tuning.
- **FlexiCT on the official LUNA25 whole CTs:** [rebuilt from the original scans](results/20261019_flexict_luna25/README.md).
  - The FLARE crops are native-voxel blocks of those scans. Two of the crops are corrupted, neither in the development fold.
  - The rebuilt pipelines reproduce the rows above: 0.9215 and 0.9170.
  - Wider context does not raise AUROC: an 80 mm field, whole-CT tokens in FlexiCT's official 1.5 mm and 2.0 mm input form, and their fusion with the crop field were all tested.

Development data is used for configuration selection. Neural-head results average three runs with consecutive seeds from 2025 through 2027. Linear heads use deterministic fits. Saved neural predictors use seed 2025. All selected pipelines use a single field of view. TAP-CT processes that field through 12-slice windows.

AutoMSC completed its official 1,000-epoch recipe with classification labels and segmentation masks; its result and the I3D result each use one seed. AutoMSC AUROC is 0.8797 [0.8254, 0.9216]. Its paired AUROC difference against VISTA3D is −0.0143 [−0.0661, +0.0312]. The [twelve-pipeline comparison](results/20261018_all_pipeline_development/README.md) provides all metrics and 2,000 paired patient-bootstrap intervals; the [eight-pipeline comparison](results/20261017_automsc_development/README.md) retains the original six encoders and two official baselines.

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

An [automatic interpretation pilot](results/20261015_autointerp_pilot/README.md) labelled 24 features with a general vision-language model and scored the labels blind on held-out development patients. Labels for trained-encoder features separate high- from size-matched low-activation patients (median AUROC 0.83–0.90); shuffled controls yield no label. Most labels describe the same coarse contrast, a free solid nodule against an attached one, which random-weight features and measured nodule properties capture as well; the label's own feature ranks first for 3 of 22 labels. VISTA3D stage2 feature 498 (basal lung beside the diaphragm) is the clearest feature-specific label. [Matching the contrast patients also on 0–5 mm ring density and solid fraction](results/20261016_autointerp_matched/README.md) lowers the measured-property baseline (median 0.85 and 0.77 for candidate and nodule-enriched features), removes more random-weight labels than trained ones, and raises the labels whose own feature ranks first to 5 of 18; features 498 and 304 (compact nodule with a fuzzy edge) keep specific labels.

A [radiomics comparison](results/20261017_sae_radiomics/README.md) relates the seed-2025 stable features of both nodule-interpretation dictionaries to 306 PyRadiomics features of the nodule, two surrounding rings and lung-boundary contact, and measures each feature's contribution to the fixed classifier.
- **Description by radiomics:** most analysable features are only partly described (median development R² 0.16 at FMCIB layer2 and 0.20 at VISTA3D stage2). Random-weight features reach similar values, so a low R² alone does not indicate a concept beyond radiomics.
- **Features that drive the classifiers:** classifier contribution is concentrated in FMCIB layer2 3015 and VISTA3D stage2 202 (mean logit change on removal −0.40 and −0.18). Both encode nodule size together with being surrounded by aerated lung rather than abutting chest wall or mediastinum; radiomics describes about half of their variance.
- **Review list:** eight features with a confirmed malignancy association, a top-50 classifier contribution and an incomplete radiomics description are listed for physician review in the heatmap viewer.

A [radiologist-rating comparison on LIDC-IDRI](results/20261019_sae_lidc/README.md) applies both dictionaries, unchanged, to 582 independent nodules (one per patient, 3–4 readers each; crops cut with the LUNA25 rule). The rules were fixed before any LIDC activation was computed.
- **Clear features:** 7 at FMCIB layer2 and 10 at VISTA3D stage2 each have:
  - a size-adjusted association with a rating that holds on held-out patients, beyond the random-weight features and in at least 7 of 8 dictionary seeds;
  - a top-50 classifier contribution.

  Example: VISTA3D stage2 182 rises with spiculation, lobulation and malignancy and falls with margin sharpness.
- **Against the random-weight control:** VISTA3D stage2 confirms about three times as many associations per analysed feature as its random-weight control (0.54 against 0.17); FMCIB layer2 does not clearly exceed its control (0.33 against 0.26).
- **Strength:** the best features match the agreement of one radiologist with the others for subtlety and malignancy, and reach 50–80 % of it for margin, spiculation, texture and calcification.
- **Review list on LIDC:** the four LUNA25 lung-contact descriptions of the review list hold on LIDC with the same sign.

A [follow-up](results/20261020_sae_lidc_followup/README.md) looks at the clear features as a reader. Each feature gets a page of its highest-activation nodules against volume-matched low ones.
- **Reader pass:** six features show a consistent, nameable pattern: VISTA3D stage2 202, 182, 743 and 392, and FMCIB layer2 3015 and 507. Examples: solid nodules lying free in aerated lung; irregular, heterogeneous nodules; small dense calcified nodules. These go into a blinded, scored reader task.
- **Grid resolution:** the FMCIB layer1 complete-context dictionary (4 mm grid) reaches the agreement of one radiologist with the others for margin (0.46 against 0.45; 0.24 at the 7 mm layer2 grid). The 12 mm VISTA3D stage3 grid shows little.
- **Dictionary probes:** a few dictionary features together exceed the best single feature, and the raw channels exceed the dictionary for margin, texture and calcification. Spiculation and lobulation stay weak at every site.

### FlexiCT dictionaries for interpretation

[Eight-seed dictionaries](results/20261021_flexict_sae/README.md) on three FlexiCT representations of the official LUNA25 CTs. Each set passes the fixed-classifier check of its deployed head, and each has far more stable features than a random-weight control.
- **FlexiCT-3D-VLM whole CT, final block 16** (the text-aligned layer): k64 with the token loss. 1,063–1,108 stable features against 10–18 for the random-weight control. With every patch token reconstructed, the CT-RATE image embedding keeps cosine 0.9994–0.9995.
- **FlexiCT-3D 32 mm field, block 10**: k32. 752–793 stable features against 137–157.
- **FlexiCT-3D-VLM 40 mm field, block 8**: k64, 192,000 training steps. 596–657 stable features against 169–192. Its stable features explain less of the representation (0.27 of the variance, against 0.43–0.54 at the other two sites).
- **Why the two field dictionaries are fine-tuned:** their max-pooling heads read directions that ordinary SAE reconstruction serves poorly. Both field dictionaries are therefore the token-loss dictionaries fine-tuned with two terms on the error the head reads, which keeps their seed-reproducible features.
- **AP at the field sites:** it is judged on the mean over the 8 seeds, since the development AP rests on 113 malignant nodules.

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
| Dataset acquisition | [download_flare.py](scripts/data/download_flare.py), [download_lidc.py](scripts/data/download_lidc.py); [LUNA25 crop check against the official scans](scripts/data/compare_luna25_crops.py) |
| Environment setup | [bootstrap_sources.ps1](scripts/pipeline/bootstrap_sources.ps1), [setup_environment.ps1](scripts/pipeline/setup_environment.ps1), [finish_environment.ps1](scripts/pipeline/finish_environment.ps1) |
| Additional model dependencies | [prepare_frozen_heads.ps1](scripts/pipeline/prepare_frozen_heads.ps1) |
| Frozen-head comparison | [fit_frozen_heads.py](scripts/pipeline/fit_frozen_heads.py) |
| Local-field comparison | [prepare_materials.py](scripts/pipeline/prepare_materials.py) |
| nnFoundation and FlexiCT pipelines | [inputs and features](src/sclvmi/encoder_features.py), [heads](src/sclvmi/encoder_pipeline.py), [selection and comparison](scripts/pipeline/compare_encoders.py); configuration [encoder_pipeline.json](configs/encoder_pipeline.json) |
| FlexiCT on official LUNA25 CTs | [inputs, features, heads and report](src/sclvmi/flexict_luna25.py), [comparison](scripts/pipeline/compare_flexict_luna25.py); configuration [flexict_luna25.json](configs/flexict_luna25.json) |
| FlexiCT SAE dictionaries | [token extraction, training, objectives, checks, selection and report](src/sclvmi/flexict_sae.py), [public export](scripts/pipeline/export_flexict_sae.py); configuration [flexict_sae.json](configs/flexict_sae.json) |
| Image and feature inference | [material_predict.py](src/sclvmi/material_predict.py) |
| Spatial SAE training and inference | [sae.py](src/sclvmi/sae.py), [sae_predict.py](src/sclvmi/sae_predict.py) |
| Multi-layer SAE comparison | [Input preparation](src/sclvmi/sae_campaign_data.py), [resumable training](src/sclvmi/sae_campaign_train.py), [fixed-classifier evaluation](src/sclvmi/sae_campaign_evaluate.py), [candidate selection](src/sclvmi/sae_campaign_compare.py), [PCA and seed diagnostics](src/sclvmi/sae_campaign_diagnostics.py) |
| Interpretation dictionaries | [inputs and random-weight encoders](src/sclvmi/sae_context.py), [cross-seed stability](src/sclvmi/sae_interp_stability.py), [fixed-classifier checks](src/sclvmi/sae_interp_evaluate.py), [feature statistics](src/sclvmi/sae_interp_profiles.py), [selection](src/sclvmi/sae_interp_select.py), [nodule-region pools](src/sclvmi/sae_nodule_pool.py) and [coverage](src/sclvmi/sae_interp_coverage.py), [feature candidates](src/sclvmi/sae_interp_candidates.py) and [their follow-up](src/sclvmi/sae_interp_followup.py), [automatic interpretation](src/sclvmi/sae_autointerp.py) with [model calls and validation](src/sclvmi/sae_autointerp_model.py), [radiomics](src/sclvmi/sae_radiomics.py) with [lung-boundary contact](src/sclvmi/nodule_context.py) and [concept comparison, classifier effects and review list](src/sclvmi/sae_concepts.py), [LIDC radiologist-rating comparison](src/sclvmi/sae_lidc.py) with [configuration](configs/sae_lidc.json) and [follow-up](src/sclvmi/sae_lidc_followup.py), [whole-CT maps](local_tools/viewers/feature_explorer/interpretation_whole.py) |
| Official baseline adapters | [luna25.py](src/sclvmi/luna25.py), [automsc.py](src/sclvmi/automsc.py) |
| AutoMSC and combined development comparison | [compare_automsc.py](scripts/pipeline/compare_automsc.py) |
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

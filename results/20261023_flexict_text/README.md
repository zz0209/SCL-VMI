# FlexiCT text readout and SAE feature interpretation (run 20261023_flexict_text)

This run asks whether the FlexiCT-3D-VLM text encoder can name the SAE features of the production dictionaries in `results/20261021_flexict_sae`. Every test, threshold and naming rule is in `configs/flexict_text.json`. It was written before the corresponding results; two rules were added later, and the config records why. The code is `src/sclvmi/flexict_text.py`. The test partition is unused.

## Data

- **LUNA25** (one nodule per patient, the radiomics profile members): 1,264 train and 422 development patients. Each is analysed in the official whole-CT window (2.0 mm, 160³, the CT-RATE evaluation preprocessing). All 3,534 train and development windows were encoded.
- **LIDC-IDRI** (independent, radiologist ratings): 582 patients with 3 or 4 readers, split into discovery and confirmation halves of 291. Windows were built from the DICOM series with the same preprocessing.
- **Text**: 44 concepts with three phrasings each, in the official CT-RATE template ("X ." against "No X."). They are the 18 CT-RATE classes, 11 nodule descriptors, 7 anatomical terms and 8 controls (conditions outside every chest CT, such as ankle fracture or cataract).
- **Scores**:
  - The scan-level score uses the official image embedding.
  - The token-level score of a final-layer patch token `p_t` is `<W_patch p_t, d_c>`. Averaged over the 8,000 tokens and added to the class-token term, it gives the unnormalised scan score exactly.

## Step 1: which concepts the text space reads (`text_tests.csv`, `usable_concepts.csv`, `scan_tests.png`, `token_tests.png`)

- **Pass rule.** A test passes on a split when it has the expected sign, its 95% patient bootstrap interval excludes 0, and it exceeds the largest control. It must pass on both splits of its dataset. A concept is usable at a level when all of its tests at that level pass. 14 of 99 tests pass.
- **Scan level, usable concepts:**
  - large nodule (log volume: LUNA25 0.48/0.48, LIDC 0.51/0.44);
  - ground-glass nodule (size-adjusted mean HU in LUNA25; LIDC texture);
  - part-solid nodule (LIDC);
  - lobulated nodule (LIDC);
  - emphysema (PD15 adjusted for image noise: LUNA25 −0.28/−0.29, LIDC −0.42/−0.32);
  - medical material (metal volume, LUNA25 0.36/0.36).
- **Scan level, one dataset only:**
  - malignancy passes in LUNA25; LIDC fails on the confirmation half.
  - spiculation passes in LIDC; LUNA25 sphericity fails.
- **Scan level, fails:** calcification, pleural contact, solid texture and the high-attenuation concepts.
- **Token level.** Only "pulmonary fibrotic sequela" (against the token's high-attenuation fraction) passes. The eight controls move together:
  - contrast between nodule tokens and far lung tokens: 0.2–1.3 SD;
  - correlation with medial position: 0.28–0.51.

  The token projection therefore mainly carries a component shared by every present-minus-absent direction. Nodule descriptors, emphysema and anatomy do not exceed it, and "lung base" and "chest wall" even have the opposite sign to the one expected.
- **Scores relative to the mean control** (`*_specific`, added after these results). With the shared component removed, only medical material (scan and token level) and part-solid (scan level) pass.

## Step 2: text names for the 1,102 stable w20_b16 features (`w20_naming.csv`)

- **Rule.** A feature is named only if all of these hold:
  - its write direction `W_patch W_dec[f]` lies in the top 1% for the concept and exceeds every control direction;
  - its activation is associated with the concept score in train and development (scan route), or co-located with the concept's token score (token route), beyond every control;
  - the same holds in at least 6 of the 7 other seeds.
- **Result.** No feature is named. Three reference-seed features pass before the cross-seed condition (two emphysema, one pulmonary fibrotic sequela).
- **Why the scores stay low.**
  - Feature write directions have cosine at most 0.27 with any concept direction.
  - Removing one feature changes the scan score of any of the seven candidate concepts by at most 0.08 SD.
  - Most stable features are active in almost every window.
- **Measured properties still separate some features.** For example:
  - a feature whose window activation follows metal volume (0.36/0.40) writes toward "medical material" (cosine 0.248) but slightly less than toward one control (0.260);
  - a feature that follows PD15 and lung volume agrees across 4 seeds.

## Step 5: field sites through w20_b16 (`bridge_f32_b10.csv`, `bridge_f40_b08.csv`)

- No field feature inherits a name, because no w20 feature is named.
- Many field features have a w20 counterpart with Spearman ≥ 0.5 on train and development:
  - f32_b10: 60 of 752 features, matching 35 distinct w20 features;
  - f40_b08: 102 of 615 features, matching 52 distinct w20 features.

## Step 4: which features the classifiers use (`contributions_*.csv`)

The effect of a feature is the head logit change on the 1,285 development nodules when its decoded contribution is removed from the read tokens.

| Site | AUROC | Remove all stable features | Remove all other features | Largest single feature |
|---|---|---|---|---|
| w20_b16 | 0.899 | 0.865 | 0.863 | mean \|Δlogit\| 0.51, seed-specific |
| f32_b10 | 0.921 | 0.913 | 0.852 | 2.25, seed-specific |
| f40_b08 | 0.917 | 0.897 | 0.781 | 2.11, seed-specific |

- The largest single effect at each site comes from a feature that reappears (decoder cosine ≥ 0.7) in only 2–3 of the 7 other seeds. Its best counterparts in those seeds have decoder cosine 0.50–0.92 and activation correlation 0.52–0.90. These three features also follow nodule volume (Spearman 0.39–0.48).
- Head probabilities recomputed from the locally encoded tokens match the stored predictions within 1.8e-3. The encoders ran on a different GPU from the one that produced the stored predictions.

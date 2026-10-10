# Dataset and SAE heatmap viewers

The dataset browser displays locally available FLARE-AutoMSC datasets, including
LUNA25. The feature explorer displays saved nodule-crop and whole-CT SAE outputs.
All viewer surfaces share the same English/Chinese globe control. Language choice
persists across reloads and is shared across these localhost services.

## Storage and dependencies

Run commands from the repository root. Use Python 3.12. Copy
`configs/storage.example.json` to `configs/storage.local.json` and fill in the
paths for your local storage. The dataset revision is recorded in
`configs/download_flare.json`. Keep your downloaded dataset in its recorded
snapshot directory and follow its access and usage terms.

For the dataset browser, copy `local_tools/network.example.json` to
`local_tools/network.local.json`. The example serves localhost. Install
`local_tools/viewers/dataset_browser/requirements.txt` in the `dataset_browser`
environment under the configured `environments` directory.

For heatmaps, install `local_tools/viewers/feature_explorer/requirements.txt` and
the repository package (`python -m pip install -e .`) in the `pipeline`
environment. Saved model artifacts and analysis outputs are required locally.

## Dataset browser

```powershell
.\local_tools\viewers\dataset_browser\start_browser.ps1
```

Open [the dataset browser](http://127.0.0.1:8765) after startup. The server runs
in the foreground; Ctrl+C stops it. Select a dataset, case and image channel to
view slices, labels, native geometry, region volumes and intensity distributions.
The LUNA25 partition display identifies the source fold grouping used by this
browser. Research runs retain their own explicit split records.

## Heatmap browser

```powershell
.\local_tools\viewers\feature_explorer\start.ps1
```

Open [heatmaps](http://127.0.0.1:8773/whole). Choose image scope, SAE configuration
and seed in the existing three-plane viewer. The six selected configuration
families provide seeds 2025, 2026 and 2027. Each dictionary retains its own Feature
selection. Feature IDs are dictionary-local; the catalog uses fixed 80-item pages,
with search, ranking and a direct numeric selector. URLs preserve the selection
and support browser Back and Forward.

Nodule crops cover 128 development patients. Spatial maps offer native samples
and interpolation. Global dictionaries return one value for the entire crop;
linked CT browsing remains available and spatial overlay controls are disabled.
The default color maximum is fixed across these 128 patients. The strongest-case
button opens the patient with the largest saved response for the current Feature.

Whole CT uses eight provider-training PETWB scans in discovery–confirmation pairs.
Selected dictionaries use training-sized windows at 32 mm intervals for FMCIB and
48 mm for VISTA3D. Spatial maps retain central native positions: 4/8 mm for FMCIB
layer1/layer2 and 6/12 mm for VISTA3D stage2/stage3. Regional modes show a central
sample, mean or maximum. Global maps show one response per input window. Window
context can influence neighboring responses. The color maximum uses four
discovery scans and remains fixed across cases.

The Observations tab opens evidence with its original dictionary, Feature and
case. Its FMCIB layer1/k64 and VISTA3D stage2/k80 dictionaries remain selectable
under the observation group. FMCIB observation maps use continuous convolutions;
selected-dictionary maps use training-sized windows. The viewer shows dictionary
identity and the relevant sampling explanation in Help.

All three planes retain physical axis spacing. Click to position the crosshair;
use the wheel, arrow keys or slice slider to navigate. Shift-drag pans a plane;
double-click resets image size. Hold Space over an image to hide the overlay.
Each plane can expand independently. CT windows, opacity, threshold and color
maximum control display. Dictionary checks expose saved development metrics and
the exact run/weight identity. The page clears old images during a new request and
provides explicit loading, preparation and retry states.

The campaign named in `configs/sae_campaign.json` must provide `viewer/index.json`,
the referenced `runs/<run_id>/dictionary.pt`, `viewer/<model>/cases.json`, the
corresponding `case_*.h5` inputs and `viewer/responses/<run_id>/responses.h5`.
Native activation caches stay at their configured campaign locations.

Selected whole-CT outputs use the run named in `configs/sae_whole_viewer.json`:
`maps/<run_id>/case_*.h5`, completion receipts and per-model completion records.
Preparation can be resumed with:

```powershell
python local_tools/viewers/feature_explorer/selected_whole.py --model fmcib
python local_tools/viewers/feature_explorer/selected_whole.py --model vista
```

For parallel output processing, use `selected_whole_fast.py --model <model>
--batch-size 16 --writers 3`. It computes the same window geometry and dictionary
responses, converts output to contiguous float16 on the GPU, and overlaps GPU
work with three independent HDF5 writers. Completed files and saved tile positions
are reused. Each execution records its source hash, batch size, writer count and
parent asset identity under `executions/`; completion receipts identify the
execution that finished each case. Floating-point differences near an SAE
threshold should be checked when changing batch size or execution environment.

The commands require the published encoder checkpoints registered in local
storage, selected dictionaries and the source audit below. Run each model's
`--smoke` option and `check_selected_whole.py --model <model>` before extraction
in a new environment. Large extractions require available GPU memory and storage.

## Feature experiments

Open [Feature experiments](http://127.0.0.1:8773/experiments) for the saved FMCIB
layer1/k64 study: 2,048 features and 128 development patients. Controls include
position templates, matched seeds, input changes and saved interventions.

The configured `runs` directory must contain the corresponding artifacts:

- `20261008_spatial_sae/`: the dictionaries referenced by the feature screen.
- `20261008_feature_validation/fmcib/`: screen, case arrays, case geometry and
  discovery spatial statistics.
- `20261009_feature_exploration/`: members, feature descriptors, rankings,
  analysis, probe maps and summaries, intervention records, `findings.json`,
  `interpretations.json`, their `findings.en.json` and `interpretations.en.json`
  translations, and the per-case `crop_browser_cache` HDF5 files.

Private findings and interpretation files travel with their analysis assets.
These files and the patient records are excluded from the source repository.

## Whole-CT source and observation assets

Required local outputs under `runs/20261009_whole_ct_exploration/` include
`source_audit.json`, completed extraction receipts and HDF5 files in
`continuous_fmcib/fmcib/` and `vista_windows/vista/`, complete analysis catalogs
and findings (including `complete_analysis/findings.en.json`), and the viewer
caches referenced by those artifacts. Source CT
paths in the private audit must remain accessible.

`start.ps1` reuses a healthy service. CT values and activation arrays remain in
configured local storage; the browser performs slice rendering without changing
saved scientific values. The viewers consume completed research assets and do
not run model training.

Patient records and derived arrays stay in private local storage. Existing
`/dictionaries?...` bookmarks open the same selection in the crop scope of the
heatmap browser. Historical `/?feature=...` bookmarks open Feature experiments.

## Raw-image SAE inference

`SpatialSAEPipeline(dictionary_path).extract(image_path)` returns the processed
input, feature activations and geometry. Spatial output includes a native-grid
RAS affine. Global output has shape `[features, 1, 1, 1]` and a null grid affine.
Global classifier checks replace the channel mean and retain the original spatial
residual; the returned probability documents this specific replacement.

Select dictionary paths from the completed private campaign catalog. Run:

```powershell
python -m sclvmi.sae_predict --dictionary <dictionary.pt> --image <input.nii.gz> --output <response.npz>
```

The command saves the activation array and a companion geometry JSON file,
including original and reconstructed classifier probabilities.

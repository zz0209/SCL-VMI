# Dataset and SAE heatmap viewers

The dataset browser displays locally available FLARE-AutoMSC datasets, including
LUNA25. The heatmap browser displays saved nodule and whole-CT responses of the
interpretation SAE dictionaries. All viewer surfaces share the same English/Chinese globe control. Language choice
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

Open [heatmaps](http://127.0.0.1:8773/whole). Choose image scope (nodule
interpretation materials or whole CT), SAE configuration and seed in the
three-plane viewer. Each dictionary retains its own Feature selection. Feature IDs are dictionary-local; the catalog uses fixed 80-item pages,
with search and ranking. Enter a Feature ID in search and press Enter to jump.
Focus the list and use Up/Down to follow its current order across pages;
the selected item scrolls into view. Arrow keys in the images navigate slices.
URLs preserve the selection
and support browser Back and Forward.

The **Nodule interpretation materials** scope shows the interpretation dictionaries
of the campaign named in `configs/sae_interpretation.json`, each with eight seeds,
and their random-weight encoder controls. It covers 256 patients: 128 training
patients (D, discovery) and 128 development patients (C, confirmation), one nodule
record each, with malignancy label and maximum diameter in the case list. FMCIB
layer1 uses an 84 mm nodule-centered input and displays only positions whose
35-voxel receptive field lies inside the source field of view; the other sites use
the deployed input windows. FMCIB layer2 (expansion 4, k8, three quarters of each
training batch from positions within 10 mm of training nodules) and VISTA3D stage2
(expansion 4, k16, half of each batch from those positions, four times longer
training) serve nodule interpretation; their stable features cover the nodule
region ([results](../../results/20261015_sae_nodule_coverage/README.md)). FMCIB
layer1 and VISTA3D stage3 keep uniformly sampled training (smallest k passing the
fixed-classifier checks, expansion 4) and are marked as not used for nodule
interpretation. The default CT window is the lung window, and a green
outline shows the nodule mask and can be hidden. The catalog marks each feature as stable, partially reappearing, unstable
or inactive from decoder matching across seeds, and can rank by reappearance,
nodule-region enrichment, size-adjusted malignancy association, image-variable
R² and activation frequency. **Feature evidence** lists the matched feature in
every other seed (open it directly), mean response by distance to the nodule
surface, development AUROC, size-adjusted partial correlations in both patient
groups, and percentiles relative to the random-weight encoder. Dictionary checks
report input positions, training-batch composition, training steps, FVU,
matched-rank PCA FVU, stable features, variance explained by stable features,
PW-MCC and fixed-classifier preservation.

This scope requires, under the configured campaign directory, `viewer/index.json`,
`viewer_members.csv`, `nodule_references.csv`, `viewer_inputs/<source>/*.h5`,
`cases/<model>/*.npz` with their geometry JSON files, the referenced
`runs/<run_id>/dictionary.pt`, `stability/<group>/` and `profiles/<model>_<site>/`.
Deployed trained activations are read from the previous suite's case caches.

Whole CT uses eight provider-training PETWB scans in discovery–confirmation pairs
and the same interpretation dictionaries: trained-encoder seeds 2025, 2026 and
2027 and the random-weight control seed 2025 of each site. Each dictionary is
applied to its training-sized window on a regular grid: FMCIB layer1 complete
context uses 84 mm windows every 48 mm and keeps positions whose 35-voxel
receptive field lies inside the CT volume; FMCIB layer2 uses 50 mm windows every
32 mm; VISTA3D uses 72 mm windows every 48 mm. Spatial maps keep the central native
positions (4, 8, 6 and 12 mm). Regional modes show a central sample, mean or
maximum. Window context can influence neighboring responses. The color maximum
uses the four discovery scans and stays fixed across cases. The catalog keeps the
cross-seed stability tags and can rank by lung and liver enrichment (discovery
mean over the region divided by the mean over other soft tissue) and by the share
of response in edge air and high-HU regions; Feature evidence and dictionary
checks are the same as in the nodule scope. FMCIB layer2 and VISTA3D stage2 browser
files come from the run named in `whole_ct.run_id`; FMCIB layer1 and VISTA3D stage3
files come from the earlier run named in `whole_ct.reused_caches`, whose
dictionaries are unchanged. A dictionary becomes available once its eight browser
files have arrived. The eight source scans are listed in
`runs/20261009_whole_ct_exploration/source_audit.json`; the CT paths recorded
there must remain accessible.

All three planes retain physical axis spacing. Click to position the crosshair;
use the wheel, arrow keys or slice slider to navigate. Shift-drag pans a plane;
double-click resets image size. Hold Space over an image to hide the overlay.
Each plane can expand independently. CT windows, opacity, threshold and color
maximum control display. Dictionary checks expose saved development metrics and
the exact run/weight identity. The page clears old images during a new request and
provides explicit loading, error and retry states.

Whole-CT maps for the interpretation dictionaries use the `whole_ct` section of
`configs/sae_interpretation.json` and are written under the run it names:

```powershell
python local_tools/viewers/feature_explorer/interpretation_whole.py --stage maps --model fmcib --sites layer2
python local_tools/viewers/feature_explorer/interpretation_whole.py --stage maps --model vista --sites stage2
python local_tools/viewers/feature_explorer/interpretation_whole.py --stage caches --workers 6
python local_tools/viewers/feature_explorer/check_interpretation_whole.py --model fmcib --cases 0,1
```

`--limit N` processes N windows per case for a smoke run in a separate directory.
Map files record their completed windows and resume from them. The cache stage
writes one gzip-compressed float16 feature per chunk, checks every copied value,
and stores per-feature sums over the lung, liver, other-tissue, high-HU and
edge-air regions used for ranking. The check script recomputes sampled positions
from the CT, encoder and dictionary independently and compares them with the maps.
The random-weight encoder is rebuilt from its initialization seed and its
parameter hash is checked against the configuration. Run `check_interpretation_api.py`
after browser files arrive.

`start.ps1` reuses a healthy service. CT values and activation arrays remain in
configured local storage; the browser performs slice rendering without changing
saved scientific values. The viewers consume completed research assets and do
not run model training. Patient records and derived arrays stay in private local
storage.

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

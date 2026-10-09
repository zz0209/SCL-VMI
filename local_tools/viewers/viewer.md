# Dataset and SAE heatmap viewers

The dataset browser displays locally available FLARE-AutoMSC datasets, including
LUNA25. The feature explorer displays saved nodule-crop and whole-CT SAE outputs.
All three surfaces share the same English/Chinese globe control. Language choice
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

## Nodule-crop heatmaps

```powershell
.\local_tools\viewers\feature_explorer\start.ps1
```

Open [crop heatmaps](http://127.0.0.1:8773/). The current saved exploration covers
2,048 FMCIB features and 128 development patients. Controls include native and
interpolated maps, position templates, matched seeds, input changes and saved
interventions. The page shows loading and request-error states explicitly.

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

## Whole-CT heatmaps

The same feature service provides [whole-CT heatmaps](http://127.0.0.1:8773/whole).
It uses saved FMCIB and VISTA3D responses from eight complete CT scans. Select
native spatial responses or single-point, mean or maximum values per sampled
region. The observations page links evidence to specific feature/case selections.

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

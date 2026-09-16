# SCL-VMI

**Sparse Concept Learning in Volumetric Medical Imaging**

SCL-VMI explores sparse concept representations in volumetric medical imaging, with an initial focus on the [FLARE-AutoMSC dataset](https://huggingface.co/datasets/FLARE-MedFM/FLARE-AutoMSC). The research scope includes CT, multi-sequence MRI, and PET/CT.

The project is at an early research stage. This repository currently provides dataset acquisition and integrity-checking utilities. Model implementations, trained checkpoints, and experimental results are not yet available.

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
python scripts/data/download_flare.py
```

On other shells, set `HF_HOME` to the same configured path before running `hf auth login` and the Python commands.

The download configuration pins repository revision `ce5ea76cde8870805c37614d7ea9b08319b3575e`: 29,059 files totaling 81,744,497,845 bytes (approximately 76.13 GiB), including the published validation directory. Allocate at least 100 GiB for acquisition; derived representations require additional space. The download does not include unpublished test data.

The downloader obtains authenticated file hashes, downloads the complete snapshot without subset filters, and checks every file against its expected size and digest. Re-running the command resumes interrupted transfers. Only a fully verified snapshot receives a `complete` status. Verification errors are reported without automatically deleting files.

After the authenticated inventory has been saved, an offline verification pass is available:

```bash
python scripts/data/download_flare.py --verify-only
```

Acquiring all subsets does not define a training split. Preserve the provider's validation role and establish patient-level grouping, longitudinal grouping, and modality pairing before model development. See the dataset provider for access conditions and subset licenses. No dataset files are redistributed by this repository.

## Tests

```bash
python scripts/data/test_download_integrity.py
```

These offline tests cover content hashes, redacted metadata, manifest mismatches, and path containment. They do not replace full-snapshot verification.

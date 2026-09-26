$ErrorActionPreference = 'Stop'
Set-Location (Resolve-Path "$PSScriptRoot/../..")
$storage = Get-Content -Raw configs/storage.local.json | ConvertFrom-Json
$python = Join-Path $storage.environments 'pipeline/Scripts/python.exe'
$env:HF_HOME = $storage.hf_home
$env:HF_HUB_CACHE = $storage.hf_hub_cache
$env:HF_XET_CACHE = $storage.hf_xet_cache
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'core.longpaths'
$env:GIT_CONFIG_VALUE_0 = 'true'
& $python -u scripts/pipeline/model_assets.py download --model coralbay
if ($LASTEXITCODE -ne 0) { throw 'CoralBay download failed' }
& $python -u scripts/pipeline/model_assets.py download --model tapct
if ($LASTEXITCODE -ne 0) { throw 'TAP-CT download failed' }
& $python -m pip install -r configs/environments/frozen_heads_requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Environment installation failed' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Environment dependency check failed' }

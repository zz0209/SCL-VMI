$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$storage = Get-Content -Raw (Join-Path $workspace 'configs/storage.local.json') | ConvertFrom-Json
$env:PIP_CACHE_DIR = Join-Path $storage.project_storage 'cache/pip'
$env:TEMP = $storage.temp_root
$env:TMP = $storage.temp_root
$python = Join-Path $storage.environments 'pipeline/Scripts/python.exe'
$spec = Get-Content -Raw (Join-Path $workspace 'configs/environments/pipeline.json') | ConvertFrom-Json
& $python -m pip install @($spec.additional_packages)
if ($LASTEXITCODE -ne 0) { throw 'Additional dependencies failed' }
& $python -m pip install -e $workspace
if ($LASTEXITCODE -ne 0) { throw 'Project package failed' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency failed' }
& $python -m pip freeze | Set-Content (Join-Path $workspace 'configs/environments/pipeline.lock.txt')

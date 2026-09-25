param([Parameter(Mandatory=$true)][string]$BasePython)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$storage = Get-Content -Raw (Join-Path $workspace 'configs/storage.local.json') | ConvertFrom-Json
$spec = Get-Content -Raw (Join-Path $workspace 'configs/environments/pipeline.json') | ConvertFrom-Json
$env:PIP_CACHE_DIR = Join-Path $storage.project_storage 'cache/pip'
$env:TEMP = $storage.temp_root
$env:TMP = $storage.temp_root
$environment = Join-Path $storage.environments 'pipeline'
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR,$env:TEMP | Out-Null
if (-not (Test-Path (Join-Path $environment 'Scripts/python.exe'))) {
    & $BasePython -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
$python = Join-Path $environment 'Scripts/python.exe'
& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip installation failed' }
& $python -m pip install @($spec.torch_packages) --index-url $spec.torch_index
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $python -m pip install @($spec.packages)
if ($LASTEXITCODE -ne 0) { throw 'Research dependencies installation failed' }
& $python -m pip install -e (Join-Path $workspace 'third_party/automsc')
if ($LASTEXITCODE -ne 0) { throw 'AutoMSC installation failed' }
& $python -m pip freeze | Set-Content (Join-Path $workspace 'configs/environments/pipeline.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency recording failed' }

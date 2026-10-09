param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'configs/storage.local.json'))) {
    $projectRoot = Split-Path $projectRoot -Parent
    if (-not $projectRoot) { throw 'Project storage configuration was not found.' }
}
$storage = Get-Content -LiteralPath (Join-Path $projectRoot 'configs/storage.local.json') -Raw | ConvertFrom-Json
$pythonPath = Join-Path $storage.environments 'dataset_browser/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the dataset_browser environment and install local_tools/viewers/dataset_browser/requirements.txt before starting.'
}
Write-Host "Open http://127.0.0.1:$Port in your browser. Press Ctrl+C to stop."
& $pythonPath (Join-Path $PSScriptRoot 'server.py') --port $Port
if ($LASTEXITCODE -ne 0) { throw "Dataset browser exited with code $LASTEXITCODE" }

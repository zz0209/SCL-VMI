$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location -LiteralPath $projectRoot
$storage = Get-Content configs/storage.local.json -Raw | ConvertFrom-Json
$pipelinePython = Join-Path $storage.environments 'pipeline/Scripts/python.exe'
foreach ($modelName in @('fmcib', 'vista')) {
    & $pipelinePython ../.resource_manager/resource_manager.py run --resource disk-d-io --project SCL-VMI --task campaign-viewer-input --wait-sec 86400 -- $pipelinePython ../.resource_manager/resource_manager.py run --resource cpu-heavy --project SCL-VMI --task campaign-viewer-input --wait-sec 86400 -- $pipelinePython -u -m sclvmi.sae_campaign_viewer --model $modelName
    if ($LASTEXITCODE -eq 75) { exit 75 }
    if ($LASTEXITCODE -ne 0) { throw "Viewer input preparation failed: $modelName" }
}

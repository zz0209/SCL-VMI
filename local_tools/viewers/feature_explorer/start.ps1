$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'configs/storage.local.json'))) {
    $projectRoot = Split-Path $projectRoot -Parent
    if (-not $projectRoot) { throw 'Project storage configuration was not found.' }
}
Set-Location $projectRoot
$storage = Get-Content configs/storage.local.json -Raw | ConvertFrom-Json
$pythonPath = Join-Path $storage.environments 'pipeline/Scripts/python.exe'
$existing = Get-NetTCPConnection -LocalPort 8773 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    $health = Invoke-RestMethod http://127.0.0.1:8773/api/health
    if ($health.run -ne '20261009_feature_exploration') { throw 'Port 8773 belongs to another service' }
    Write-Output 'http://127.0.0.1:8773/'
    return
}
$stamp = Get-Date -Format yyyyMMddTHHmmss
$process = Start-Process -FilePath $pythonPath -ArgumentList '-u', 'local_tools/viewers/feature_explorer/server.py' -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput "$PSScriptRoot/server_$stamp.stdout.log" -RedirectStandardError "$PSScriptRoot/server_$stamp.stderr.log" -PassThru
$deadline = (Get-Date).AddSeconds(30)
while (-not (Get-NetTCPConnection -LocalPort 8773 -State Listen -ErrorAction SilentlyContinue)) {
    if ($process.HasExited) { throw "Feature explorer exited. Read server_$stamp.stderr.log." }
    if ((Get-Date) -gt $deadline) { throw "Feature explorer startup timed out. Read server_$stamp.stderr.log." }
    Start-Sleep -Milliseconds 200
}
$health = Invoke-RestMethod http://127.0.0.1:8773/api/health
if ($health.run -ne '20261009_feature_exploration') { throw 'Unexpected service on port 8773.' }
Write-Output "Feature explorer ready: http://127.0.0.1:8773/"

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$sources = @(
    @('automsc', 'https://github.com/medfm-flare/AutoMSC-Baseline', '858c26bc745d094f00c3eb03bdf795fba6588ac9'),
    @('luna25', 'https://github.com/DIAGNijmegen/luna25-baseline-public', 'eff27763470640059423a90c05dd018166ea0815'),
    @('ctfm', 'https://github.com/project-lighter/CT-FM', 'f58c89a75ff0270ba6e339027e40bf04a221735d'),
    @('vista', 'https://github.com/Project-MONAI/VISTA', 'd4a8fe0dbf5cb4b76c531fccca8e29c1e6f6ee45'),
    @('genesis', 'https://github.com/MrGiovanni/ModelsGenesis', '4c6c3bdee39e27c20622f1e9a2a8186833e97ceb'),
    @('fmcib', 'https://github.com/AIM-Harvard/foundation-cancer-image-biomarker', '1f1c0c8725c110c9c70cb466467a55e3160760c9')
)
New-Item -ItemType Directory -Force -Path (Join-Path $workspace 'third_party') | Out-Null
foreach ($source in $sources) {
    $destination = Join-Path $workspace ('third_party/' + $source[0])
    if (-not (Test-Path $destination)) {
        & git -c http.sslBackend=openssl clone --no-checkout $source[1] $destination
        if ($LASTEXITCODE -ne 0) { throw "Clone failed: $($source[0])" }
        if ($source[0] -ne 'fmcib') {
            & git -C $destination checkout --detach $source[2]
            if ($LASTEXITCODE -ne 0) { throw "Checkout failed: $($source[0])" }
        }
    }
    $reference = if ($source[0] -eq 'fmcib') { $source[2] } else { 'HEAD' }
    $actual = & git -C $destination rev-parse $reference
    if ($actual -ne $source[2]) { throw "Source version differs: $($source[0])" }
}
$fmcibSource = Join-Path $workspace 'third_party/fmcib_source'
if (-not (Test-Path $fmcibSource)) {
    $archive = Join-Path $workspace 'third_party/fmcib_source_selected.zip'
    & git -c core.protectNTFS=false -C (Join-Path $workspace 'third_party/fmcib') archive --format=zip "--output=$archive" '1f1c0c8725c110c9c70cb466467a55e3160760c9' fmcib docs experiments LICENSE README.md pyproject.toml
    if ($LASTEXITCODE -ne 0) { throw 'FMCIB source extraction failed' }
    Expand-Archive -LiteralPath $archive -DestinationPath $fmcibSource
}

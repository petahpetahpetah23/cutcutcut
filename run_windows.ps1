$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Card Cutter is not set up. Run setup_windows.ps1 first."
}

$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$env:ORT_DISABLE_TELEMETRY = "1"
& .\.venv\Scripts\python.exe -m card_cutter


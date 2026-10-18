$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    throw "Card Cutter supports 64-bit x64 Windows only in this release."
}

$probe = Join-Path $PSScriptRoot ".card-cutter-write-test"
try {
    [IO.File]::WriteAllText($probe, "test")
} finally {
    Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path "$env:WINDIR\System32\vcruntime140.dll")) {
    throw "Microsoft Visual C++ Runtime is required. Install the current x64 redistributable from Microsoft, then rerun setup."
}

$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python Launcher was not found. Install 64-bit Python 3.14 from https://www.python.org/downloads/windows/"
}

py -3.14 -c "import platform, struct, sys; assert sys.version_info[:2] == (3, 14); assert struct.calcsize('P') * 8 == 64; print('Using Python', platform.python_version())"
py -3.14 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock

if ($args -contains "-Dev") {
    & .\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-dev.lock
}

Write-Host "Setup complete. Run run_windows.ps1 to start Card Cutter."


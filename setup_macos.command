#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  print -u2 "Card Cutter supports Apple Silicon Macs only in this release."
  exit 1
fi

if [[ ! -w . ]]; then
  print -u2 "This folder is not writable. Move Card Cutter to a writable folder."
  exit 1
fi

if ! command -v python3 >/dev/null; then
  print -u2 "Python 3.14 was not found. Install it from https://www.python.org/downloads/"
  exit 1
fi

python3 - <<'PY'
import platform
import sys
if sys.version_info[:2] != (3, 14):
    raise SystemExit(f"Python 3.14 is required; found {platform.python_version()}")
PY

python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock

if [[ "${1:-}" == "--dev" ]]; then
  .venv/bin/python -m pip install --require-hashes -r requirements-dev.lock
fi

print "Setup complete. Double-click run_macos.command to start Card Cutter."


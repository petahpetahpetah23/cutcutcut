#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print -u2 "Card Cutter is not set up. Run setup_macos.command first."
  exit 1
fi
export PYTHONPATH="$PWD/src"
export ORT_DISABLE_TELEMETRY=1
exec .venv/bin/python -m card_cutter


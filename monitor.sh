#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" == */* ]]; then cd -- "${SCRIPT_PATH%/*}"; fi
source ./common.sh
resolve_python
exec "$PY" monitor.py --out "${OUT:-runs/round2}" "$@"

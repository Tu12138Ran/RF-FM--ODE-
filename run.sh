#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" == */* ]]; then cd -- "${SCRIPT_PATH%/*}"; fi
source ./common.sh
resolve_python
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec "$PY" -u run.py "$@"

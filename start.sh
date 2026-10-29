#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" == */* ]]; then cd -- "${SCRIPT_PATH%/*}"; fi
OUT="${OUT:-runs/round2}"
CONFIG="${CONFIG:-configs/round2.json}"
source ./common.sh
resolve_python
export PYTHON_BIN="$PY"
mkdir -p -- "$OUT"
nohup bash run.sh --config "$CONFIG" --out "$OUT" "$@" >> "$OUT/console.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$OUT/launcher.pid"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Startup failed; inspect $OUT/console.log" >&2
  tail -n 30 -- "$OUT/console.log"
  exit 1
fi
echo "Started PID $PID. SSH may be disconnected; training continues."
echo "Monitor: $PY monitor.py --out \"$OUT\""
echo "Log: tail -f \"$OUT/console.log\""
echo "Resume: repeat the same start command."

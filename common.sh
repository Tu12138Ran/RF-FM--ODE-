#!/usr/bin/env bash
# Sourced after changing into the project directory. Paths are read, never evaluated.
resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="$PYTHON_BIN"
  elif [[ -f .python_path ]]; then
    IFS= read -r PY < .python_path
  elif [[ -x .venv/bin/python ]]; then
    PY="$PWD/.venv/bin/python"
  elif [[ -x ../flow_path_lab/.venv/bin/python ]]; then
    PY="$PWD/../flow_path_lab/.venv/bin/python"
  else
    echo "No configured Python. Run bash setup.sh first." >&2
    return 1
  fi
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python not found: $PY. Run bash setup.sh again." >&2
    return 1
  fi
}

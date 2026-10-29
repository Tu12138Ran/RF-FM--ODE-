#!/usr/bin/env bash
set -euo pipefail
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "$SCRIPT_PATH" == */* ]]; then cd -- "${SCRIPT_PATH%/*}"; fi
usable() {
  "$1" -c 'import sys,torch,numpy,matplotlib; assert (3,10)<=sys.version_info[:2]<=(3,13); assert tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])>=(2,4); assert torch.cuda.is_available()' >/dev/null 2>&1
}

PY=""
for candidate in "${PYTHON_BIN:-}" "$PWD/.venv/bin/python" "$PWD/../flow_path_lab/.venv/bin/python"; do
  if [[ -n "$candidate" ]] && command -v "$candidate" >/dev/null 2>&1 && usable "$candidate"; then
    PY="$candidate"
    echo "Reusing working CUDA environment: $PY"
    break
  fi
done

if [[ -z "$PY" ]]; then
  BASE="${PYTHON_BIN:-python3}"
  "$BASE" -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,13), "Use Python 3.10-3.13"'
  # Repair partial virtualenvs: an existing Python binary does not imply pip exists.
  if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    if ! "$BASE" -m venv --system-site-packages .venv; then
      VERSION="$("$BASE" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      echo "Virtualenv creation failed. On Ubuntu/Debian install python${VERSION}-venv, then rerun setup.sh." >&2
      echo "Or set PYTHON_BIN to the working Python from your first-round environment." >&2
      exit 1
    fi
  fi
  PY="$PWD/.venv/bin/python"
  if ! "$PY" -c 'import torch; assert tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])>=(2,4); assert torch.cuda.is_available()' >/dev/null 2>&1; then
    "$PY" -m pip install --upgrade --force-reinstall "torch==2.7.1" --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
  fi
  "$PY" -m pip install -r requirements.txt
fi

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("PyTorch:",torch.__version__); print("GPU:",torch.cuda.get_device_name(0)); print("BF16:",torch.cuda.is_bf16_supported())'
"$PY" -m unittest discover -s tests -v
"$PY" -c 'import sys; print(sys.executable)' > .python_path
echo "Ready. Start round 2 with: CUDA_VISIBLE_DEVICES=0 bash start.sh"

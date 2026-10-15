#!/usr/bin/env bash
# Installs vLLM in WSL2 Ubuntu for Aequitas (faculty requirement: serve models with vLLM).
# Run inside Ubuntu:  bash scripts/vllm/install_wsl.sh
#
# Needs: an NVIDIA Windows driver with WSL support (nvidia-smi must work inside Ubuntu).
# Gemma 4 MTP needs a recent vLLM: older releases treat the -assistant checkpoint as a generic draft
# model and fail on multimodal targets (vllm-project/vllm#42005). Check the log for method='mtp'.
set -euo pipefail

VENV="${VENV:-$HOME/vllm-venv}"

nvidia-smi >/dev/null || { echo "nvidia-smi failed inside WSL: install/update the NVIDIA Windows driver"; exit 1; }
sudo apt-get update -y && sudo apt-get install -y python3-venv python3-dev build-essential curl

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip uv
uv pip install -U vllm --torch-backend=auto
uv pip install -U llmcompressor   # for scripts/quantize (Phase 3)

python - <<'PY'
import torch, vllm
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
echo "Done. Activate with: source $VENV/bin/activate   Then: bash scripts/vllm/serve.sh"

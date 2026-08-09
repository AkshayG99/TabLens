#!/usr/bin/env bash
# ==============================================================================
# Setup DisCO (verl PR #3357) training environment on the A6000 Linux VM.
# Run from the repo root:  bash scripts/setup_disco_a6000.sh
#
# Prereqs before running:
#   - CUDA toolkit 12.x (nvcc)  ->  nvcc --version
#   - A Python env (conda/venv) already created
#   - torch installed (cu124):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   - Model weights on this box (see MODEL_PATH note below)
# ==============================================================================
set -euo pipefail

echo "==> 0/5 Sanity checks..."
nvcc --version >/dev/null 2>&1 || { echo "ERROR: CUDA toolkit not found (needed for flash-attn). Install CUDA 12.x first."; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || { echo "ERROR: torch not installed."; exit 1; }

echo "==> 1/5 Upgrading build tools..."
pip install --upgrade pip wheel setuptools ninja packaging

echo "==> 2/5 Installing verl with DisCO (PR #3357, NOT yet merged -> use PR head)..."
if [ ! -d "verl" ]; then
    git clone https://github.com/volcengine/verl.git
fi
cd verl
git fetch origin pull/3357/head:disco
git checkout disco
pip install -e .
cd ..

echo "==> 3/5 Installing vLLM (rollout engine) + fast kernels..."
pip install vllm
pip install flash-attn --no-build-isolation
pip install liger-kernel
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention --no-build-isolation

echo "==> 4/5 Verifying DisCO recipe is importable..."
python -c "import recipe.disco.main_disco" 2>&1 | head -1 || true

echo "==> 5/5 Done."
echo
echo "Next steps (manual):"
echo "  1. Edit run_disco_qwen.sh -> export MODEL_PATH pointing at the merged model"
echo "     (either a local dir or a HuggingFace repo id like <user>/qwen3.5-9b-merged)"
echo "  2. Set n_gpus_per_node to match this VM's GPU count (run: nvidia-smi | grep -c 'A6000')"
echo "  3. bash run_disco_qwen.sh"

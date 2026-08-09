#!/usr/bin/env bash
# ==============================================================================
# Setup DisCO (verl PR #3357) training environment on the A6000 Linux VM.
# Run from the repo root:  bash scripts/setup_disco_a6000.sh
#
# Prereqs before running:
#   - NVIDIA GPU driver + nvidia-smi working
#   - A Python env (conda/venv) already created
#   - torch installed (cu124):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   - Model weights on this box (see MODEL_PATH note below)
#
# CUDA toolkit (nvcc) is only needed if flash-attn has to be built from source.
# flash-attn ships prebuilt wheels for common torch+CUDA combos, so we try the
# wheel first and fall back to a source build only if nvcc is available.
# ==============================================================================
set -euo pipefail

HAS_NVCC=false
if nvcc --version >/dev/null 2>&1; then
    HAS_NVCC=true
    echo "==> 0/5 nvcc found; source-build fallback available."
else
    echo "==> 0/5 No nvcc/CUDA toolkit. Will rely on prebuilt wheels."
    echo "     (If flash-attn install fails below, install CUDA 12.x and rerun:)"
    echo "       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "       sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update"
    echo "       sudo apt-get install -y cuda-toolkit-12-8"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || { echo "ERROR: torch not installed. Install torch (cu124) first."; exit 1; }

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

echo "==> 3b/5 Installing flash-attn (prebuilt wheel preferred)..."
if pip install flash-attn 2>/tmp/fa_err.txt; then
    echo "     flash-attn: prebuilt wheel OK."
else
    echo "     flash-attn: no prebuilt wheel for this torch/CUDA combo."
    if $HAS_NVCC; then
        echo "     Building from source with nvcc (can take 15-30 min)..."
        pip install flash-attn --no-build-isolation
    else
        echo "     ERROR: prebuilt wheel missing AND no nvcc to build from source."
        echo "     Install CUDA toolkit 12.x, then rerun:"
        echo "       pip install flash-attn --no-build-isolation"
        cat /tmp/fa_err.txt
        exit 1
    fi
fi

echo "==> 3c/5 Installing liger-kernel + linear-attn kernels..."
pip install liger-kernel
pip install causal-conv1d
pip install flash-linear-attention

echo "==> 4/5 Verifying DisCO recipe is importable..."
python -c "import recipe.disco.main_disco" 2>&1 | head -1 || true

echo "==> 5/5 Done."
echo
echo "Next steps (manual):"
echo "  1. Edit run_disco_qwen.sh -> export MODEL_PATH pointing at the merged model"
echo "     (either a local dir or a HuggingFace repo id like <user>/qwen3.5-9b-merged)"
echo "  2. Set n_gpus_per_node to match this VM's GPU count (run: nvidia-smi | grep -c 'A6000')"
echo "  3. bash run_disco_qwen.sh"

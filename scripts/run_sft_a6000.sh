#!/usr/bin/env bash
# Run optimized TabLens SFT training on an RTX A6000 (48GB) Linux VM.
# Installs flash-attn + liger-kernel (and Qwen3.5's linear-attn kernels), then
# trains with configs/sft_config_a6000.yaml (batch 8, packed, no padding waste).
#
# Requirements before first run: CUDA toolkit matching your torch build for nvcc.
#   nvcc --version        # must be 12.x if torch is cu124
#   python -c "import torch; print(torch.__version__, torch.version.cuda)"
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> 1/3 Installing build deps..."
pip install --upgrade pip wheel setuptools ninja packaging

echo "==> 2/3 Installing fast kernels (one-time, flash-attn takes ~15-30 min to build)..."
# flash-attn must be built against the installed torch -> --no-build-isolation
pip install flash-attn --no-build-isolation
# liger-kernel (pure triton, prebuilt wheel)
pip install liger-kernel
# Qwen3.5 hybrid model: 24/32 layers are linear attention -> these are the
# real throughput win (flash-attn alone only helps the 8 dense layers).
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention --no-build-isolation

echo "==> 3/3 Starting SFT training..."
echo "Dataset: data/cot_datasets/german_v2_cot_train.jsonl"
python -m llamafactory.cli train configs/sft_config_a6000.yaml

echo
echo "Training complete. Model saved to outputs/qwen3.5-9b-sft-a6000"

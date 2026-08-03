#!/bin/bash
set -e

# Reduce CUDA memory fragmentation (avoids OOM near peak, per PyTorch guidance).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Triton-kernel deps for faster SFT: liger (fused CE/RoPE/MLP/RMSNorm) +
# FlashAttention-3. If the flash-attn build fails on your GPU, set
# flash_attn: sdpa in configs/sft_config.yaml (liger still gives most of the win).
pip install -r LLaMA-Factory/requirements/liger-kernel.txt
pip install "flash-attn>=2.7"

echo "Starting SFT training with Qwen3.5-9B-Instruct (unsloth backend)..."
echo "Dataset: data/cot_datasets/german_v2_cot_train.jsonl"
llamafactory-cli train configs/sft_config.yaml

echo "Training complete. Model saved to outputs/qwen3.5-9b-sft"

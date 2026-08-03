#!/bin/bash
set -e

# Reduce CUDA memory fragmentation (avoids OOM near peak, per PyTorch guidance).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting SFT training with Qwen3.5-9B-Instruct (unsloth backend)..."
echo "Dataset: data/cot_datasets/german_v2_cot_train.jsonl"
llamafactory-cli train configs/sft_config.yaml

echo "Training complete. Model saved to outputs/qwen3.5-9b-sft"

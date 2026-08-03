#!/usr/bin/env bash
# Run TabLens SFT training on a Linux VM with an RTX A6000 (48GB).
# Uses the A6000-tuned config: batch 4 x grad-accum 2, no flash-attn/liger needed.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Starting SFT training with Qwen3.5-9B (unsloth backend)..."
echo "Dataset: data/cot_datasets/german_v2_cot_train.jsonl"
python -m llamafactory.cli train configs/sft_config_a6000.yaml

echo
echo "Training complete. Model saved to outputs/qwen3.5-9b-sft-a6000"

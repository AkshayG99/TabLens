#!/bin/bash
# run via ./eval_qwen.sh [adapter_path] [dataset]
# example: ./eval_qwen.sh outputs/qwen3.5-9b-sft-a6000/checkpoint-450 german
set -e

source .venv/bin/activate

ADAPTER_PATH="${1:-outputs/qwen3.5-9b-sft-a6000/checkpoint-450}"
DATASET="${2:-german}"

echo "Evaluating adapter: $ADAPTER_PATH (dataset: $DATASET)"

python eval/evaluate.py \
    --dataset "$DATASET" \
    --adapter-path "$ADAPTER_PATH"

#!/bin/bash
# GRPO RL fine-tuning - VM profile (A6000 48GB, bf16 LoRA).
# Usage:
#   bash run_grpo_a6000.sh                      # full run (1 epoch)
#   bash run_grpo_a6000.sh --max-steps 3 ...    # smoke test first!
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$HOME/tablens-venv/bin/python3" ]; then
    PYTHON="$HOME/tablens-venv/bin/python3"
else
    PYTHON="python3"
fi

MODEL_PATH="${MODEL_PATH:-creativelapse/qwen3.5-9b-merged}"

mkdir -p logs
"$PYTHON" rl_grpo/train_grpo.py \
    --profile a6000 \
    --model "$MODEL_PATH" \
    "$@" 2>&1 | tee "logs/grpo_a6000_$(date +%Y%m%d_%H%M%S).log"

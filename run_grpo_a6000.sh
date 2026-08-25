#!/bin/bash
# GRPO RL fine-tuning wrapper.
# Profile is auto-detected from GPU VRAM by train_grpo.py
#   (a6000 >=38GB bf16 / 24g >=20GB 4-bit QLoRA / 4070 <20GB).
# Force one explicitly with e.g.: bash run_grpo_a6000.sh --profile a6000
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
    --model "$MODEL_PATH" \
    "$@" 2>&1 | tee "logs/grpo_a6000_$(date +%Y%m%d_%H%M%S).log"

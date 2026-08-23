#!/bin/bash
# GRPO RL fine-tuning - RTX 4070 12GB local profile (forces 4-bit QLoRA).
# Usage:   bash run_grpo_4070.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$HOME/tablens-venv/bin/python3" ]; then
    PYTHON="$HOME/tablens-venv/bin/python3"
else
    PYTHON="python3"
fi

# Prefer the locally merged model; fall back to the HF repo id.
MODEL_PATH="$SCRIPT_DIR/outputs/qwen3.5-9b-merged"
[ -d "$MODEL_PATH" ] || MODEL_PATH="creativelapse/qwen3.5-9b-merged"

mkdir -p logs
"$PYTHON" rl_grpo/train_grpo.py \
    --profile 4070 \
    --model "$MODEL_PATH" \
    "$@" 2>&1 | tee "logs/grpo_4070_$(date +%Y%m%d_%H%M%S).log"

#!/bin/bash
# Universal GRPO launcher - THE one to use on any server box.
# - profile auto-detected from GPU VRAM inside train_grpo.py
#   (a6000 >=38GB bf16 / 24g >=20GB 4-bit QLoRA / 4070 <20GB)
# - never trusts a stale exported MODEL_PATH pointing at a folder
#   this machine doesn't have; falls back to the HF hub model.
#
# Usage:
#   bash run_grpo.sh                                  # full run, foreground
#   bash run_grpo.sh --max-steps 3 --train-limit 20   # smoke test first!
#   nohup bash run_grpo.sh > /dev/null 2>&1 &         # backgrounded (log still written)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$HOME/tablens-venv/bin/python3" ]; then
    PYTHON="$HOME/tablens-venv/bin/python3"
else
    PYTHON="python3"
fi

HF_DEFAULT="creativelapse/qwen3.5-9b-merged"
MODEL_PATH="${MODEL_PATH:-$HF_DEFAULT}"
case "$MODEL_PATH" in
    outputs/*|/*|~/*)
        if [ ! -d "$MODEL_PATH" ]; then
            echo "[run_grpo] WARNING: '$MODEL_PATH' is not a folder on this machine."
            echo "[run_grpo]          Falling back to HF hub: $HF_DEFAULT"
            MODEL_PATH="$HF_DEFAULT"
        fi
        ;;
esac

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs
LOG="logs/grpo_run_$(date +%Y%m%d_%H%M%S).log"

echo "[run_grpo] gpu   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'n/a')"
echo "[run_grpo] model : $MODEL_PATH"
echo "[run_grpo] log   : $LOG"

"$PYTHON" rl_grpo/train_grpo.py --model "$MODEL_PATH" "$@" 2>&1 | tee "$LOG"

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
            echo "[run_disco] WARNING: '$MODEL_PATH' is not a folder on this machine."
            echo "[run_disco]          Falling back to HF hub: $HF_DEFAULT"
            MODEL_PATH="$HF_DEFAULT"
        fi
        ;;
esac

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m pip show causal-conv1d >/dev/null 2>&1 || "$PYTHON" -m pip install causal-conv1d --no-build-isolation || \
    echo "[run_disco] [warn] causal-conv1d build failed -- continuing without it, this only costs speed"
"$PYTHON" -m pip show flash-linear-attention >/dev/null 2>&1 || "$PYTHON" -m pip install flash-linear-attention --no-build-isolation || \
    echo "[run_disco] [warn] flash-linear-attention build failed -- continuing without it, this only costs speed"

mkdir -p logs
LOG="logs/disco_run_$(date +%Y%m%d_%H%M%S).log"

echo "[run_disco] gpu   : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 || echo 'n/a')"
echo "[run_disco] model : $MODEL_PATH"
echo "[run_disco] log   : $LOG"

"$PYTHON" rl_grpo/train_grpo.py --model "$MODEL_PATH" --disco "$@" 2>&1 | tee "$LOG"

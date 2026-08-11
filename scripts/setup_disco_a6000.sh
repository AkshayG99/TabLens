#!/usr/bin/env bash
# ==============================================================================
# Setup DisCO (verl PR #3357) training environment on the A6000 Linux VM.
# Run from the repo root:  bash scripts/setup_disco_a6000.sh
#
# Prereqs before running:
#   - NVIDIA GPU driver + nvidia-smi working
#   - A Python env (conda/venv) already created
#   - torch installed (cu124):  pip install torch --index-url https://download.pytorch.org/whl/cu124
#   - Model weights on this box (see MODEL_PATH note below)
#
# CUDA toolkit (nvcc) is only needed if flash-attn has to be built from source.
# flash-attn ships prebuilt wheels for common torch+CUDA combos, so we try the
# wheel first and fall back to a source build only if nvcc is available.
#
# This script ALSO applies the four compatibility patches required by the
# transformers/ray/vllm versions pip resolves today. A fresh install will NOT
# train without them:
#   A) transformers 5.x:  AutoModelForVision2Seq -> AutoModelForImageTextToText
#   B) transformers 5.x:  vllm get_cached_tokenizer (all_special_tokens_extended)
#   C) Python 3.12/uvloop: fsdp_workers.py event-loop guard
#   D) Ray dashboard_agent: opentelemetry/protobuf version pins
# All patches are idempotent - safe to rerun.
# ==============================================================================
set -euo pipefail

HAS_NVCC=false
if nvcc --version >/dev/null 2>&1; then
    HAS_NVCC=true
    echo "==> 0/6 nvcc found; source-build fallback available."
else
    echo "==> 0/6 No nvcc/CUDA toolkit. Will rely on prebuilt wheels."
    echo "     (If flash-attn install fails below, install CUDA 12.x and rerun:)"
    echo "       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
    echo "       sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update"
    echo "       sudo apt-get install -y cuda-toolkit-12-8"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" || { echo "ERROR: torch not installed. Install torch (cu124) first."; exit 1; }

echo "==> 0b/6 Installing tmux (so training survives SSH disconnects)..."
if command -v tmux >/dev/null 2>&1; then
    echo "     tmux already installed ($(tmux -V))"
else
    if [ "$(id -u)" -eq 0 ]; then
        apt-get update -qq && apt-get install -y tmux
    else
        sudo apt-get update -qq && sudo apt-get install -y tmux
    fi
    echo "     installed $(tmux -V)"
fi

echo "==> 1/6 Upgrading build tools..."
pip install --upgrade pip wheel setuptools ninja packaging

echo "==> 2/6 Installing verl with DisCO (PR #3357, NOT yet merged -> use PR head)..."
if [ ! -d "verl" ]; then
    git clone https://github.com/volcengine/verl.git
fi
cd verl
git fetch origin pull/3357/head:disco
git checkout disco
pip install -e .
cd ..

echo "==> 2a/6 Patch A: transformers 5.x rename (AutoModelForVision2Seq -> AutoModelForImageTextToText)..."
if python -c "import transformers as t; raise SystemExit(0 if int(t.__version__.split('.')[0]) >= 5 else 1)" 2>/dev/null; then
    sed -i 's/AutoModelForVision2Seq/AutoModelForImageTextToText/' \
        verl/verl/workers/fsdp_workers.py \
        verl/verl/utils/model.py \
        verl/verl/utils/checkpoint/fsdp_checkpoint_manager.py \
        verl/verl/model_merger/base_model_merger.py \
        verl/scripts/legacy_model_merger.py
    if grep -rq 'AutoModelForVision2Seq' verl/verl verl/scripts; then
        echo "     WARNING: some AutoModelForVision2Seq references remain"
    else
        echo "     OK"
    fi
else
    echo "     transformers <5 detected; rename not needed, skipping"
fi

echo "==> 2b/6 Patch C: fsdp_workers.py event-loop guard (Python 3.12 + uvloop)..."
python - <<'PYEOF'
p = "verl/verl/workers/fsdp_workers.py"
s = open(p).read()
old = "            loop = asyncio.get_event_loop()"
new = ('            try:\n'
       '                loop = asyncio.get_event_loop()\n'
       '            except RuntimeError:\n'
       '                loop = asyncio.new_event_loop()\n'
       '                asyncio.set_event_loop(loop)')
if old in s and "asyncio.new_event_loop()" not in s:
    open(p, "w").write(s.replace(old, new))
    print("     patched event-loop guard")
else:
    print("     already patched or pattern changed; skipping")
PYEOF

echo "==> 3/6 Installing vLLM (rollout engine) + fast kernels..."
pip install vllm

echo "==> 3a/6 Patch B: vllm get_cached_tokenizer (transformers 5.x removed all_special_tokens_extended)..."
python - <<'PYEOF'
import os, re, vllm
p = os.path.join(os.path.dirname(vllm.__file__), "transformers_utils", "tokenizer.py")
s = open(p).read()
s2 = re.sub(r"tokenizer\.all_special_tokens_extended",
            'getattr(tokenizer, "all_special_tokens_extended", None)', s)
if s2 != s:
    open(p, "w").write(s2)
    print("     patched vllm tokenizer")
else:
    print("     already patched; skipping")
PYEOF

echo "==> 3b/6 Installing flash-attn (prebuilt wheel preferred)..."
if pip install flash-attn 2>/tmp/fa_err.txt; then
    echo "     flash-attn: prebuilt wheel OK."
else
    echo "     flash-attn: no prebuilt wheel for this torch/CUDA combo."
    if $HAS_NVCC; then
        echo "     Building from source with nvcc (can take 15-30 min)..."
        pip install flash-attn --no-build-isolation
    else
        echo "     ERROR: prebuilt wheel missing AND no nvcc to build from source."
        echo "     Install CUDA toolkit 12.x, then rerun:"
        echo "       pip install flash-attn --no-build-isolation"
        cat /tmp/fa_err.txt
        exit 1
    fi
fi

echo "==> 3c/6 Installing liger-kernel + linear-attn kernels..."
pip install liger-kernel
pip install causal-conv1d
pip install flash-linear-attention

echo "==> 3d/6 Patch D: pin opentelemetry/protobuf (Ray dashboard_agent crash fix)..."
# Deliberately ignores vllm's metadata pins; works fine at runtime.
pip install --no-deps \
    opentelemetry-api==1.44.0 \
    opentelemetry-sdk==1.44.0 \
    opentelemetry-proto==1.44.0 \
    opentelemetry-exporter-prometheus==0.65b0 \
    opentelemetry-semantic-conventions==0.65b0 \
    protobuf==7.35.1

echo "==> 4/6 Verifying DisCO recipe is importable + patches applied..."
python -c "import recipe.disco.main_disco" 2>&1 | head -1 || true

echo "     -- patch checks --"
if grep -q 'AutoModelForVision2Seq' verl/verl/workers/fsdp_workers.py 2>/dev/null; then
    echo "     transformers rename:   MISSING"
else
    echo "     transformers rename:   OK"
fi
if python -c "import vllm, os; p=os.path.join(os.path.dirname(vllm.__file__),'transformers_utils','tokenizer.py'); s=open(p).read(); raise SystemExit(0 if 'all_special_tokens_extended\", None)' in s else 1)" 2>/dev/null; then
    echo "     vllm tokenizer:        OK"
else
    echo "     vllm tokenizer:        MISSING"
fi
if python -c "s=open('verl/verl/workers/fsdp_workers.py').read(); raise SystemExit(0 if 'asyncio.new_event_loop()' in s else 1)" 2>/dev/null; then
    echo "     event-loop guard:      OK"
else
    echo "     event-loop guard:      MISSING"
fi
if python -c "import importlib.metadata as m; raise SystemExit(0 if m.version('opentelemetry-sdk').startswith('1.44') else 1)" 2>/dev/null; then
    echo "     otel/protobuf pins:    OK"
else
    echo "     otel/protobuf pins:    MISSING"
fi

echo "==> 5/6 Done."
echo
echo "Next steps (manual):"
echo "  1. Edit run_disco_qwen.sh -> export MODEL_PATH pointing at the merged model"
echo "     (either a local dir or a HuggingFace repo id like <user>/qwen3.5-9b-merged)"
echo "  2. Set n_gpus_per_node to match this VM's GPU count (run: nvidia-smi | grep -c 'A6000')"
echo "  3. Start a tmux session so training keeps running if you disconnect:"
echo "       tmux new -s train"
echo "  4. Run training (venv MUST be active):"
echo "       bash run_disco_qwen.sh 2>&1 | tee /tmp/disco_smoke.log"
echo "     Detach with Ctrl-b d, reattach with: tmux attach -t train"

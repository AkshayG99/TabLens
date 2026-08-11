# DisCO / verl RL Training on the VM — Actually-Works Setup

This branch carries the fixes that make DisCO (verl PR #3357) RL training run end-to-end.
Follow this top-to-bottom on a fresh Linux VM with an NVIDIA GPU. It was validated by
running all 50 steps of a smoke test (Qwen2.5-0.5B) cleanly to completion.

## 1. VM requirements

- Ubuntu 22.04, NVIDIA driver installed, `nvidia-smi` works
- GPU: A6000 48GB recommended (the committed config also works on a 12GB card)
- Host RAM: 32GB+ (see `optimizer_offload` note in Troubleshooting)
- Disk: 60GB+ (model weights + checkpoint cache)

## 2. Clone + branch

```bash
git clone https://github.com/AkshayG99/TabLens.git
cd TabLens
git checkout a6000-optimized
```

## 3. Python env + torch (cu124)

```bash
python3 -m venv ~/tablens-venv
source ~/tablens-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

## 4. Install verl + kernels

```bash
bash scripts/setup_disco_a6000.sh
```

This installs verl (PR #3357 head), vLLM, flash-attn, liger-kernel, causal-conv1d,
flash-linear-attention **and auto-applies the four compatibility patches in
section 5** (idempotent, safe to rerun; it verifies them at the end). It clones
verl into `./verl` inside the repo; `run_disco_qwen.sh` uses that location (or
`~/verl`) automatically.

## 5. Compatibility patches (now automatic)

A fresh install pulls transformers 5.x and the latest Ray, which break verl/vLLM.
`scripts/setup_disco_a6000.sh` applies all four below; skip this section unless
you set the environment up by hand.

### 5a. transformers 5.x rename (`AutoModelForVision2Seq` -> `AutoModelForImageTextToText`)

```bash
cd TabLens/verl
sed -i 's/AutoModelForVision2Seq/AutoModelForImageTextToText/' \
  verl/workers/fsdp_workers.py \
  verl/utils/model.py \
  verl/utils/checkpoint/fsdp_checkpoint_manager.py \
  verl/model_merger/base_model_merger.py \
  scripts/legacy_model_merger.py
grep -rn AutoModelForVision2Seq verl scripts || echo "OK: no remaining references"
```

### 5b. vLLM tokenizer patch (transformers v5 removed `all_special_tokens_extended`)

Verified against vLLM 0.8.4:

```bash
python - <<'EOF'
import os, re, vllm
p = os.path.join(os.path.dirname(vllm.__file__),
                 "transformers_utils", "tokenizer.py")
s = open(p).read()
s2 = re.sub(r"tokenizer\.all_special_tokens_extended",
            'getattr(tokenizer, "all_special_tokens_extended", None)', s)
assert s2 != s, "pattern not found - check vllm version / already patched?"
open(p, "w").write(s2)
print("patched", p)
EOF
```

### 5c. Python 3.12 + uvloop event-loop fix (`fsdp_workers.py`)

```bash
cd TabLens/verl
python - <<'EOF'
p = "verl/workers/fsdp_workers.py"
s = open(p).read()
old = "            loop = asyncio.get_event_loop()"
new = ('            try:\n'
       '                loop = asyncio.get_event_loop()\n'
       '            except RuntimeError:\n'
       '                loop = asyncio.new_event_loop()\n'
       '                asyncio.set_event_loop(loop)')
assert old in s, "sync-mode init line not found"
open(p, "w").write(s.replace(old, new))
print("patched fsdp_workers.py")
EOF
```

### 5d. Ray / opentelemetry version pins

Ray's dashboard_agent crashes with the versions pip resolves by default. Pin the
known-good set. `--no-deps` is intentional — this set violates vLLM's metadata pins
but works fine at runtime:

```bash
pip install --no-deps \
  opentelemetry-api==1.44.0 \
  opentelemetry-sdk==1.44.0 \
  opentelemetry-proto==1.44.0 \
  opentelemetry-exporter-prometheus==0.65b0 \
  opentelemetry-semantic-conventions==0.65b0 \
  protobuf==7.35.1
```

Verify Ray starts and sees the GPU:

```bash
ray start --head
ray status     # expect GPUs: 1
ray stop
```

## 6. Data

The repo already contains correctly formatted parquets (`data/train.parquet`,
`data/val.parquet`). Only re-run this if you change the source JSONL:

```bash
python prepare_data.py
```

verl's `RLHFDataset` requires three columns — `prepare_data.py` emits exactly these:
do NOT re-format `prompt` into a tokenized string (that was the original bug):

- `prompt` — list of `{"role": ..., "content": ...}` messages (system + user)
- `data_source` — e.g. `tablens/german_credit` (selects the reward fn)
- `reward_model` — `{"style": "rule", "ground_truth": "0"|"1"}`

## 7. Configure and run

Edit `run_disco_qwen.sh`:

- `MODEL_PATH` — your merged model (HF repo id or local dir). For a first end-to-end
  check, keep the default `Qwen/Qwen2.5-0.5B`.
- `n_gpus_per_node` — match the VM (`nvidia-smi | grep -c A6000`).

The committed batch sizes / memory knobs are the known-good **12GB-GPU** settings
(50/50 steps clean). On an A6000 (48GB) you can scale up: `train_batch_size`,
`ppo_mini_batch_size`, `max_num_batched_tokens=8192`, `max_num_seqs=256`,
`gpu_memory_utilization=0.85`, `ppo_max_token_len_per_gpu`.

**Run it with the venv ACTIVE** (otherwise `python3` is the system interpreter and
you get `ModuleNotFoundError: No module named 'hydra'`):

```bash
cd TabLens
source ~/tablens-venv/bin/activate
bash run_disco_qwen.sh 2>&1 | tee /tmp/disco_smoke.log
```

## 8. What success looks like

- `Size of train dataloader: 50, Size of val dataloader: 4`
- vLLM engine init lines, then `GPU KV cache size: ...`
- `Training Progress:` ticking at ~10s/it
- Custom reward firing every step: `[ground_truth] 0` / `[score] 0.0`
- Clean exit with `Final validation metrics: {'val-core/tablens/german_credit/reward/mean@1': ...}`
- Checkpoint saved under `checkpoints/verl-disco/...`

## 9. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'hydra'` | venv not activated before running |
| `Total available GPUs 0 is less than total desired GPUs 1` | stale Ray session. `rm -rf /tmp/ray`, `ray stop --force`, kill leftover `ray`/`main_disco` procs, rerun |
| `No available memory for the cache blocks` | vLLM's profiled peak exceeds `gpu_memory_utilization × total_vram`. Lower `max_num_seqs` and/or raise `gpu_memory_utilization` |
| `Task main_task failed due to oom` | host RAM exhausted. `optimizer_offload=True` holds the Adam state in RAM. Add RAM/swap, or move optimizer back to GPU if VRAM allows |
| CUDA `out of memory at cumem_allocator.cpp` | both offloads False with high `gpu_memory_utilization` on a small card. Use `param_offload=False` + `optimizer_offload=True` |
| wandb `ImportError: cannot import name 'Imports'` | broken wandb/protobuf combo — logger is already `['console']`, leave it |
| All rewards 0.0, `pg_loss` 0.0 | expected with an untrained base model — it never emits ACCEPT/REJECT. Use the SFT-merged model as `MODEL_PATH`, or relax `reward/credit_reward.py` for partial credit |

## 10. Reward signal (why smoke-test rewards are 0)

`reward/credit_reward.py` returns `1.0` only when the response contains an
`ACCEPT`/`REJECT` verdict matching `ground_truth`, else `0.0`. A random 0.5B model
never produces a verdict, so every reward is 0 and there is no learning signal.
The smoke test proves the pipeline works; for real training point `MODEL_PATH` at
the merged 9B SFT model (which already emits verdicts) or make the reward denser.

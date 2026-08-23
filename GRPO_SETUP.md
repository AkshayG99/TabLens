# TabLens RL — TRL GRPO setup (replaces verl/DisCO)

The old RL path (verl PR #3357 + DisCO + Ray + Hydra + vLLM 0.8.4) is retired.
It was an unmerged research PR requiring four hand-applied source patches,
hand-pinned opentelemetry (`--no-deps`), a Ray cluster and FSDP sharding — all
to train LoRA on **one** GPU. It also ran with `rollout_n=1`, which gives
group-based advantage estimation nothing to contrast, i.e. no learning signal
even when it ran. Do not resurrect it.

## What replaced it

- **TRL `GRPOTrainer`** (already in `requirements.txt`, trl==0.24.0). Single
  python process: transformers + peft + datasets only. No new dependencies.
- **Dense reward** (`rl_grpo/rewards.py`): no-verdict = 0.0, wrong verdict =
  0.05, correct = 1.0 (×1.5 for REJECT ground truth, the minority class).
- **LoRA on top of the merged SFT model** (`creativelapse/qwen3.5-9b-merged`),
  same as SFT used, so the policy already emits parseable verdicts.

## Setup (fresh VM)

```bash
git clone https://github.com/AkshayG99/TabLens.git && cd TabLens
git checkout <this branch>
python3 -m venv ~/tablens-venv && source ~/tablens-venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt        # includes trl, peft, bitsandbytes
nvidia-smi                             # GPU visible? then you're done.
```

No CUDA toolkit install, no flash-attn build, no patches, no ray/hydra.

## Smoke test first (minutes, not hours)

```bash
source ~/tablens-venv/bin/activate
bash run_grpo_a6000.sh --max-steps 3 --train-limit 64 --save-steps 1000
```

Success looks like:

- `[env] torch ... | gpu NVIDIA A6000`
- `[baseline] pre-train rewards: [...]` — non-zero values mean the merged
  model emits verdicts and the reward parses them. If these are ALL 0.0, stop:
  the model/prompt/reward triangle is broken; do not start a real run.
- `[reward] ...` style training metrics ticking per step (`rewards/mean` rising
  over steps is the actual learning signal)
- adapter saved to `outputs/grpo/<model>-<profile>/`

## Real run

```bash
# VM (48GB): bf16 LoRA r32, effective batch 4 prompts x 8 gens
nohup bash run_grpo_a6000.sh > /dev/null 2>&1 &   # logs/grpo_a6000_*.log

# Local 12GB card: 4-bit QLoRA r16, 4 prompts x 4 gens
bash run_grpo_4070.sh
```

Defaults live in `PROFILES` inside `rl_grpo/train_grpo.py`; every value is
overridable by flags (`--num-generations --per-device-batch --grad-accum
--lora-rank --lr --epochs ...`). Constraint:
`per_device_batch % num_generations == 0`.

## Tuning knobs that matter

| Symptom | Knob |
|---|---|
| OOM | lower `--per-device-batch` (keep it a multiple of generations); on 48GB also try `--quantize 4bit` |
| rewards flat at 0 | check `[baseline]` block; raise `--minority-weight`; verify prompts look right in logs |
| degenerate one-word answers | they score 0.05 vs 1.0 for correct — signal is fine, keep training; or lower temp via config |
| too slow | `pip install vllm` if your vLLM supports qwen3.5, add `--use-vllm` (colocate mode) |

## After training

The output dir holds a PEFT adapter (not merged weights). Merge it the same way
as the SFT one (`merge_lora.py`), then evaluate with `eval/evaluate.py`.

#!/usr/bin/env python3
"""TabLens RL stage: GRPO fine-tuning with TRL. Replaces the verl/DisCO stack.

Single process, single GPU, no Ray, no Hydra, no source patches.
LoRA/QLoRA via PEFT, dense rule-based reward from rl_grpo/rewards.py.

Examples:
    # Any single-GPU box: profile auto-detected from VRAM
    #   a6000 (>=38 GB, bf16 LoRA) / 24g (>=20 GB, 4-bit QLoRA) / 4070 (<20 GB)
    python rl_grpo/train_grpo.py

    # Or pin one explicitly
    python rl_grpo/train_grpo.py --profile a6000
    python rl_grpo/train_grpo.py --profile 24g
    python rl_grpo/train_grpo.py --profile 4070

    # 10-minute smoke test before burning real compute
    python rl_grpo/train_grpo.py --max-steps 3 --num-generations 8 \
        --per-device-batch 8 --train-limit 64
"""

import argparse
import functools
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Reduce allocator fragmentation on tight cards (recommended by the CUDA OOM
# message itself); set before any torch/CUDA init. setdefault so an explicit
# export still wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl.import_utils as _trl_import_utils

# trl==0.24.0 bug: optional-dependency flags store (bool, version) tuples, so
# every is_<pkg>_available() check is truthy even when the package is MISSING
# (crashes the grpo_trainer import chain: mergekit, llm_blender, ...). Recompute
# each flag as a real bool before anything in trl reads it.
def _pkg_installed(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

for _attr in [a for a in dir(_trl_import_utils) if a.startswith("_") and a.endswith("_available")]:
    setattr(_trl_import_utils, _attr,
            _pkg_installed(_attr[1:-len("_available")]))

from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards import credit_reward_fn, extract_verdict, score_completion  # noqa: E402
from prompts import SYSTEM_PROMPT  # noqa: E402
from disco_trainer import DiscoGRPOTrainer  # noqa: E402

PROFILES = {
    "a6000": dict(  # RTX A6000 48GB / similar 40-48GB cards
        quantize="4bit",
        lora_rank=32,
        lora_alpha=64,
        num_generations=8,
        per_device_batch=16,   # completions per fwd/bwd; must be a multiple of num_generations
        grad_accum=2,          # -> effective batch 32 completions = 4 prompts x 8 gens
        max_prompt_length=1024,
        # Was 512: checkpoint-100's own trainer_state.json shows
        # completions/clipped_ratio == 1.0 and mean_terminated_length == 0.0
        # on every single logged step -- literally no rollout ever reached a
        # natural stop at 512, so (with mask_truncated_completions=True below)
        # ~every completion was masked out of the loss the whole run. SFT
        # completions average ~1100-1350 tokens before concluding, so bump
        # toward that. Watch VRAM: the lm_head logprob pass materializes
        # batch x seq_len x 248k-vocab logits, i.e. cost scales ~linearly
        # with this value. Run the smoke test first; drop back toward 512 if
        # it OOMs before a real run.
        max_completion_length=1024,
        lr=1e-5,
        optim="adamw_torch_fused",
    ),
    "24g": dict(  # 24 GB cards (3090/4090/L4/...): bf16 weights alone are ~18.6 GB,
        quantize="4bit",  # so 4-bit QLoRA is mandatory. First tuning (8 gens x batch 8)
        lora_rank=32,     # OOMed at step 0: the lm_head logprob pass materializes
        lora_alpha=64,    # batch x 512 x 248k logits on top of rollout states. 4x4x8
        num_generations=4,
        per_device_batch=4,
        grad_accum=8,          # -> effective batch 32 completions = 8 prompts x 4 gens
        max_prompt_length=1024,
        # Was 512, same zero-natural-termination problem as the a6000 profile
        # above. Smaller bump than a6000 (768 vs 1024) since this card already
        # OOMed once at the wider rollout width above and has less headroom
        # for the lm_head logprob pass; tune down further if it OOMs.
        max_completion_length=768,
        lr=1e-5,
        optim="adamw_bnb_8bit",
    ),
    "4070": dict(  # 12-16GB cards: 4-bit QLoRA mandatory
        quantize="4bit",
        lora_rank=16,
        lora_alpha=32,
        # 2 gens x batch 2: the lm_head logprob pass materializes
        # batch x seq x 248k logits; batch 4 spiked past 12 GB into WDDM
        # system-memory spill (nvlddmkm hangs / 10x slowdowns).
        num_generations=2,
        per_device_batch=2,
        grad_accum=8,          # -> effective batch 16 completions = 8 prompts x 2 gens
        max_prompt_length=512,
        # Was 384, same zero-natural-termination problem, smallest bump of
        # the three profiles -- this card has the least VRAM headroom.
        max_completion_length=512,
        lr=1e-5,
        # NOT paged_*: bnb's own tests skip paged optimizers on win32
        # (unified-memory paging hangs WDDM at teardown).
        optim="adamw_bnb_8bit",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="creativelapse/qwen3.5-9b-merged")
    p.add_argument("--profile", choices=["auto"] + sorted(PROFILES), default="auto",
                   help="VRAM tier; 'auto' detects from the GPU (default)")
    p.add_argument("--quantize", choices=["none", "4bit"], default=None,
                   help="Override profile quantization")
    p.add_argument("--train-file", default="data/processed/german_v2/train.jsonl")
    p.add_argument("--val-file", default="data/processed/german_v2/val.jsonl")
    p.add_argument("--eval", action="store_true",
                   help="Opt in to periodic validation. OFF by default: a GRPO eval pass "
                   "generates num_generations completions for EVERY val row, same cost as a "
                   "train step -- one pass over the full 100-row val set took >2h in one run, "
                   "and with the default --eval-steps 25 that fires before most runs even "
                   "reach --save-steps 50. Use --val-limit to shrink it if you opt in.")
    p.add_argument("--output-dir", default=None,
                   help="Default: outputs/grpo/<model-name>-<profile>")
    p.add_argument("--train-limit", type=int, default=None, help="Cap train rows (smoke tests)")
    p.add_argument("--val-limit", type=int, default=None,
                   help="Cap val rows -- strongly recommended if you pass --eval at all")
    p.add_argument("--eval-steps", type=int, default=25,
                   help="Run validation every N steps; eval_reward shows up in the logs for you "
                   "to read, but is NOT used for automatic checkpoint selection (no "
                   "load_best_model_at_end -- GRPOTrainer's custom reward metrics never reach "
                   "the metrics dict Trainer's own best-checkpoint logic reads, so wiring that "
                   "up is a guaranteed KeyError, not a maybe). Pick the best checkpoint "
                   "yourself from the logged eval_reward values.")
    p.add_argument("--minority-weight", type=float, default=1.5,
                   help="Reward multiplier when ground truth is REJECT (minority class)")
    # Hyperparameter overrides (rarely needed; profiles are tuned already)
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--per-device-batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--disco", action="store_true")
    p.add_argument("--disco-score-func", choices=["logL", "Lratio"], default="logL",
                   help="logL: mean log-likelihood under the current policy. Lratio: mean "
                   "importance ratio vs the old policy. Paper recommends tau=10 for logL, "
                   "tau=1 for Lratio -- change --disco-tau to match if you switch this.")
    p.add_argument("--disco-delta", type=float, default=1e-4,
                   help="Target KL-divergence ceiling for DisCO's constraint term.")
    p.add_argument("--disco-beta", type=float, default=1e3,
                   help="DisCO's adaptive constraint penalty weight -- NOT the same thing as "
                   "--beta (TRL's linear GRPO KL coefficient); the two are independent "
                   "hyperparameters from different papers that happen to share a name. Not "
                   "validated for this task -- the paper tuned this for math reasoning "
                   "benchmarks, treat it as a starting point.")
    p.add_argument("--disco-tau", type=float, default=10.0,
                   help="Temperature for DisCO's soft-max aggregation over negative responses' "
                   "scores within a group.")
    p.add_argument("--max-completion-length", type=int, default=None,
                   help="Override profile's rollout token budget. Directly trades off against "
                   "both memory (lm_head logprob pass scales with this) and wall-clock "
                   "(shorter generations finish faster) -- the two things you're usually "
                   "tuning for when nothing else is left to cut.")
    p.add_argument("--max-prompt-length", type=int, default=None,
                   help="Override profile's prompt token budget.")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-vllm", action="store_true",
                   help="Colocated vLLM generation (only if installed vLLM supports this arch)")
    return p.parse_args()


class ZeroLossGuard(TrainerCallback):
    """Aborts training if loss stays exactly 0.0 for too many consecutive
    logged steps -- the signature of every rollout getting clipped by
    max_completion_length and then masked out of the loss. checkpoint-100's
    trainer_state.json showed loss == 0.0 on all 100 of its logged steps,
    meaning it received ~zero effective gradient the entire run, discovered
    only after the fact by reading the log post-hoc. Fail fast instead: a
    real GRPO loss is a continuous float, so several exact 0.0s in a row this
    early isn't noise.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self._zero_streak = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero or logs is None or "loss" not in logs:
            return
        self._zero_streak = self._zero_streak + 1 if logs["loss"] == 0.0 else 0
        if self._zero_streak >= self.patience:
            raise RuntimeError(
                f"loss has been exactly 0.0 for {self._zero_streak} consecutive logged "
                f"steps -- almost certainly every rollout is being clipped by "
                f"max_completion_length and masked out of the loss. Check "
                f"completions/clipped_ratio and completions/mean_terminated_length in "
                f"the logs above; if clipped_ratio is ~1.0, raise max_completion_length "
                f"or make the model's reasoning shorter via the prompt before retrying."
            )


def build_dataset(path, limit=None):
    ds = load_dataset("json", data_files=path, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ds = ds.map(
        lambda row: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["text"]},
            ],
            "target": int(row["target"]),
        },
        remove_columns=[c for c in ds.column_names if c not in ("text",)],
    )
    return ds


def main():
    args = parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - check driver / venv (nvidia-smi first).")
    if args.profile == "auto":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 38:
            args.profile = "a6000"      # 48GB-class: bf16 LoRA fits
        elif vram_gb >= 20:
            args.profile = "24g"        # 24GB-class: bf16 weights alone are too big
        else:
            args.profile = "4070"
        print(f"[env] profile auto -> '{args.profile}' ({vram_gb:.1f} GB VRAM)")
    prof = dict(PROFILES[args.profile])
    if args.quantize is not None:
        prof["quantize"] = args.quantize
    for key, arg in [("num_generations", "num_generations"),
                     ("per_device_batch", "per_device_batch"),
                     ("grad_accum", "grad_accum"),
                     ("lora_rank", "lora_rank"),
                     ("lora_alpha", "lora_alpha"),
                     ("lr", "lr"),
                     ("max_completion_length", "max_completion_length"),
                     ("max_prompt_length", "max_prompt_length")]:
        override = getattr(args, arg)
        if override is not None:
            prof[key] = override

    per_device = prof["per_device_batch"]
    n_gens = prof["num_generations"]
    if per_device % n_gens != 0:
        raise SystemExit(
            f"per-device batch ({per_device}) must be divisible by num_generations ({n_gens})"
        )
    print(f"[env] torch {torch.__version__} | cuda {torch.version.cuda} | gpu {torch.cuda.get_device_name(0)}")

    output_dir = args.output_dir or f"outputs/grpo/{os.path.basename(args.model.rstrip('/'))}-{args.profile}"
    train_ds = build_dataset(args.train_file, args.train_limit)
    print(f"[data] {len(train_ds)} train rows from {args.train_file}")
    val_ds = None
    if args.eval:
        val_ds = build_dataset(args.val_file, args.val_limit)
        print(f"[data] {len(val_ds)} val rows from {args.val_file}")

    model_init_kwargs = {"torch_dtype": "bfloat16"}
    if prof["quantize"] == "4bit":
        from transformers import BitsAndBytesConfig
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if args.disco and n_gens < 2:
        raise SystemExit(
            f"--disco needs --num-generations >= 2 (got {n_gens}). A group of 1 can never "
            f"contain both a correct and incorrect response, so DisCO's pos/neg contrast has "
            f"nothing to work with -- this is the exact misconfiguration that gave the original "
            f"verl DisCO attempt zero learning signal. See rl_grpo/disco_loss.py's docstring."
        )

    liger_available = _pkg_installed("liger_kernel") and not args.disco
    if args.disco:
        print("[env] --disco set -- using DiscoGRPOTrainer (rl_grpo/disco_trainer.py), "
              "use_liger_loss forced off (no DisCO-flavored fused kernel exists)")
    elif liger_available:
        print("[env] liger_kernel found -- using LigerFusedLinearGRPOLoss "
              "(avoids materializing full-vocab logits, the actual OOM cause)")
    else:
        print("[env] liger_kernel NOT installed -- falling back to the standard GRPO loss. "
              "`pip install liger-kernel` (pure triton, no build) to fix the largest memory "
              "spike in this profile; SFT already depends on it successfully on this hardware.")

    cfg_kwargs = dict(
        output_dir=output_dir,
        seed=args.seed,
        # Generation (rollout)
        num_generations=n_gens,
        max_prompt_length=prof["max_prompt_length"],
        max_completion_length=prof["max_completion_length"],
        temperature=1.0,
        top_p=1.0,
        # Was True. SYSTEM_PROMPT now asks for the verdict as the FIRST line,
        # so a truncated completion (didn't reach EOS within
        # max_completion_length) almost always still contains a scorable
        # verdict -- only the reasoning tail got cut off. Masking it out of
        # the loss entirely (the old behavior) discarded that reward signal
        # even when the completion was correct, which combined with
        # max_completion_length historically being far below the model's
        # natural response length meant most batches got ~zero effective
        # gradient (see checkpoint-100's trainer_state.json: clipped_ratio
        # 1.0 and loss 0.0 on every one of its 100 logged steps).
        mask_truncated_completions=False,
        beta=args.beta,
        use_liger_loss=liger_available,
        # Optimization
        learning_rate=prof["lr"],
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        optim=prof["optim"],
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=prof["grad_accum"],
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.2,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        log_completions=False,
    )
    if val_ds is not None:
        # Monitoring only -- do NOT wire load_best_model_at_end/
        # metric_for_best_model here. GRPOTrainer.log() merges its custom
        # reward stats ("eval_reward" etc.) into what gets PRINTED/logged,
        # but that merge happens on a local dict inside log() and is never
        # written back into output.metrics -- the dict Trainer.evaluate()
        # actually returns and that _determine_best_metric indexes into.
        # GRPOTrainer doesn't override evaluate() at all (checked against
        # trl==0.24.0 source), so that dict only ever has the base keys
        # (eval_loss, eval_runtime, ...). metric_for_best_model="eval_reward"
        # was previously set here and is a guaranteed KeyError on every run
        # that includes eval -- confirmed live after a ~2h eval pass crashed
        # with exactly that, discarding the run with no checkpoint saved.
        # eval_reward still appears in the console/log output during
        # training for you to read; there's just no automatic "pick the best
        # one" -- you do that manually from the logs after the run.
        cfg_kwargs.update(
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=per_device,
        )
    if args.use_vllm:
        cfg_kwargs.update(
            use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.25
        )

    cfg = GRPOConfig(**cfg_kwargs)

    # Load here instead of passing a path: transformers 5.x no longer creates
    # PreTrainedModel.warnings_issued eagerly, but trl 0.24 writes into it
    # (grpo_trainer.py:406) after PEFT wrapping, where the missing attribute is
    # fatal. Seed it while we still hold the raw model. GRPOTrainer still
    # applies peft_config to a passed model instance (prepare_peft_model).
    lm = AutoModelForCausalLM.from_pretrained(args.model, **model_init_kwargs)
    if not hasattr(lm, "warnings_issued"):
        lm.warnings_issued = {}

    # credit_reward_fn accepts minority_weight but GRPOTrainer calls
    # reward_funcs with only (completions, prompts=..., **dataset_columns) --
    # bind --minority-weight here via a thin wrapper rather than a bare
    # functools.partial (TRL reads reward_funcs[i].__name__ for metric keys
    # like "rewards/credit_reward_fn/mean"; a partial object has no
    # __name__ and would crash trainer construction).
    def reward_fn(completions, prompts=None, target=None, **kwargs):
        return credit_reward_fn(completions, prompts=prompts, target=target,
                                 minority_weight=args.minority_weight, **kwargs)
    reward_fn.__name__ = "credit_reward_fn"

    trainer_cls = DiscoGRPOTrainer if args.disco else GRPOTrainer
    trainer_kwargs = dict(
        model=lm,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # Text-only model: pass the tokenizer explicitly. Without this, TRL
        # calls AutoProcessor, which resolves to the multimodal Qwen3.5
        # processor and demands a preprocessor_config.json we don't ship.
        processing_class=AutoTokenizer.from_pretrained(args.model),
        peft_config=LoraConfig(
            r=prof["lora_rank"],
            lora_alpha=prof["lora_alpha"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
        callbacks=[ZeroLossGuard()],
    )
    if args.disco:
        trainer_kwargs.update(
            disco_score_func=args.disco_score_func,
            disco_delta=args.disco_delta,
            disco_beta=args.disco_beta,
            disco_tau=args.disco_tau,
        )
    trainer = trainer_cls(**trainer_kwargs)

    # peft.prepare_model_for_kbit_training upcasts every non-quantized param
    # to fp32. Two problems on a 12 GB card: (1) qwen3_5's linear-attention
    # conv1d must match the bf16 activations from its 4-bit projections or
    # F.conv1d crashes during generation, and (2) the 248k-vocab embedding
    # doubles to ~4 GB, pushing VRAM into WDDM shared-memory spill -> nvlddmkm
    # hangs. Cast both back; norms stay fp32 (tiny, stability-intentional) and
    # RMSNorm re-upcasts activations internally either way.
    if prof["quantize"] == "4bit":
        for mod in trainer.model.modules():
            if isinstance(mod, torch.nn.Conv1d) and mod.weight.dtype != torch.bfloat16:
                mod.weight.data = mod.weight.data.to(torch.bfloat16)
                if mod.bias is not None:
                    mod.bias.data = mod.bias.data.to(torch.bfloat16)
            elif isinstance(mod, torch.nn.Embedding) and mod.weight.dtype != torch.bfloat16:
                mod.weight.data = mod.weight.data.to(torch.bfloat16)

    # Baseline sanity check before any training: what does the merged SFT model score?
    try:
        probe_rows = list(train_ds.select(range(min(4, len(train_ds)))))
        tok = trainer.processing_class
        texts = [
            tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
            for r in probe_rows
        ]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(trainer.model.device)
        out = trainer.model.generate(**enc, max_new_tokens=prof["max_completion_length"],
                                     do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        decoded = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        base_rewards = [score_completion(d, r["target"]) for d, r in zip(decoded, probe_rows)]
        print(f"[baseline] pre-train rewards on {len(decoded)} val-style prompts: {base_rewards}")
        for i, d in enumerate(decoded):
            print(f"[baseline] sample {i}: verdict={extract_verdict(d)} :: {d[:160]!r}")
    except Exception as e:  # noqa: BLE001 - never let diagnostics kill the run
        print(f"[baseline] skipped ({e})")

    trainer.train()
    trainer.save_model()
    if trainer.processing_class is not None:
        trainer.processing_class.save_pretrained(output_dir)
    print(f"[done] LoRA adapter saved to {output_dir}")
    print(f"[next] merge with: python merge_lora.py --adapter {output_dir} ...")


if __name__ == "__main__":
    main()

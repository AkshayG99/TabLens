#!/usr/bin/env python3
"""TabLens RL stage: GRPO fine-tuning with TRL. Replaces the verl/DisCO stack.

Single process, single GPU, no Ray, no Hydra, no source patches.
LoRA/QLoRA via PEFT, dense rule-based reward from rl_grpo/rewards.py.

Examples:
    # VM (A6000 48GB), full run
    python rl_grpo/train_grpo.py --profile a6000

    # Local 12GB card (forces 4-bit QLoRA)
    python rl_grpo/train_grpo.py --profile 4070

    # 10-minute smoke test before burning real compute
    python rl_grpo/train_grpo.py --profile a6000 --max-steps 3 --num-generations 8 \
        --per-device-batch 8 --train-limit 64
"""

import argparse
import json
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards import credit_reward_fn, extract_verdict, score_completion  # noqa: E402

SYSTEM_PROMPT = (
    "You are a credit risk analyst for a bank. Review the applicant profile below "
    "and assess whether the loan application should be approved.\n"
    "Reason step by step, concisely.\n"
    "End your response with a final line of exactly 'Final decision: ACCEPT' "
    "or 'Final decision: REJECT'."
)

PROFILES = {
    "a6000": dict(  # RTX A6000 48GB / similar 40-48GB cards
        quantize="none",
        lora_rank=32,
        lora_alpha=64,
        num_generations=8,
        per_device_batch=16,   # completions per fwd/bwd; must be a multiple of num_generations
        grad_accum=2,          # -> effective batch 32 completions = 4 prompts x 8 gens
        max_prompt_length=1024,
        max_completion_length=512,
        lr=1e-5,
        optim="adamw_torch_fused",
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
        max_completion_length=384,
        lr=1e-5,
        # NOT paged_*: bnb's own tests skip paged optimizers on win32
        # (unified-memory paging hangs WDDM at teardown).
        optim="adamw_bnb_8bit",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="creativelapse/qwen3.5-9b-merged")
    p.add_argument("--profile", choices=sorted(PROFILES), default="a6000")
    p.add_argument("--quantize", choices=["none", "4bit"], default=None,
                   help="Override profile quantization")
    p.add_argument("--train-file", default="data/processed/german_v2/train.jsonl")
    p.add_argument("--val-file", default="data/processed/german_v2/val.jsonl")
    p.add_argument("--output-dir", default=None,
                   help="Default: outputs/grpo/<model-name>-<profile>")
    p.add_argument("--train-limit", type=int, default=None, help="Cap train rows (smoke tests)")
    p.add_argument("--minority-weight", type=float, default=1.5,
                   help="Reward multiplier when ground truth is REJECT (minority class)")
    # Hyperparameter overrides (rarely needed; profiles are tuned already)
    p.add_argument("--num-generations", type=int, default=None)
    p.add_argument("--per-device-batch", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lora-rank", type=int, default=None)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-vllm", action="store_true",
                   help="Colocated vLLM generation (only if installed vLLM supports this arch)")
    return p.parse_args()


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
    prof = dict(PROFILES[args.profile])
    if args.quantize is not None:
        prof["quantize"] = args.quantize
    for key, arg in [("num_generations", "num_generations"),
                     ("per_device_batch", "per_device_batch"),
                     ("grad_accum", "grad_accum"),
                     ("lora_rank", "lora_rank"),
                     ("lora_alpha", "lora_alpha"),
                     ("lr", "lr")]:
        override = getattr(args, arg)
        if override is not None:
            prof[key] = override

    per_device = prof["per_device_batch"]
    n_gens = prof["num_generations"]
    if per_device % n_gens != 0:
        raise SystemExit(
            f"per-device batch ({per_device}) must be divisible by num_generations ({n_gens})"
        )

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - check driver / venv (nvidia-smi first).")
    print(f"[env] torch {torch.__version__} | cuda {torch.version.cuda} | gpu {torch.cuda.get_device_name(0)}")

    output_dir = args.output_dir or f"outputs/grpo/{os.path.basename(args.model.rstrip('/'))}-{args.profile}"
    train_ds = build_dataset(args.train_file, args.train_limit)
    print(f"[data] {len(train_ds)} train rows from {args.train_file}")

    model_init_kwargs = {"torch_dtype": "bfloat16"}
    if prof["quantize"] == "4bit":
        from transformers import BitsAndBytesConfig
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    cfg_kwargs = dict(
        output_dir=output_dir,
        seed=args.seed,
        # Generation (rollout)
        num_generations=n_gens,
        max_prompt_length=prof["max_prompt_length"],
        max_completion_length=prof["max_completion_length"],
        temperature=1.0,
        top_p=1.0,
        mask_truncated_completions=True,
        # Loss: no KL/ref model (same choice as the DisCO config, saves ~18GB)
        beta=0.0,
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

    trainer = GRPOTrainer(
        model=lm,
        reward_funcs=credit_reward_fn,
        args=cfg,
        train_dataset=train_ds,
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
    )

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

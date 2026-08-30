import argparse
import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import compute_all
from data.reasoning_prompt import parse_response, SYSTEM_PROMPT, DATASET_INSTRUCTIONS
from rl_grpo.prompts import SYSTEM_PROMPT as GRPO_SYSTEM_PROMPT

DEFAULT_BASE_MODEL = "unsloth/Qwen3.5-9B"
DEFAULT_ADAPTER_PATH = "outputs/qwen3.5-9b-sft"
DEFAULT_DATASET = "german"
DEFAULT_TEST_FILE = "data/processed/german_v2/test.jsonl"
RESULTS_DIR = Path("eval/results")

GERMAN_V2_INSTRUCTION = (
    "You are an expert credit risk analyst. Analyze the following German Credit applicant "
    "profile and provide a detailed, step-by-step Chain-of-Thought (COT) explanation of why "
    "this applicant should be accepted or rejected."
)

_BOLD_VERDICT = re.compile(r"\*\*(ACCEPT|REJECT)(?:ED)?\*\*", re.IGNORECASE)
_DECISION_LINE = re.compile(
    r"\b(?:final\s+|my\s+)?decision\s*:?\s*\**\s*(ACCEPT|REJECT)(?:ED)?\b", re.IGNORECASE
)
_CONCLUSION_LINE = re.compile(
    r"\b(?:conclusion|verdict|outcome|recommendation)s?\s*:?\s*\**\s*(ACCEPT|REJECT)(?:ED)?\b",
    re.IGNORECASE,
)
_VERDICT_WORD = re.compile(r"\b(ACCEPT|REJECT)(?:S|ED|ING)?\b", re.IGNORECASE)
_GOOD_BAD_WORD = re.compile(r"\b(good|bad)\b", re.IGNORECASE)

# Token IDs (Qwen3 tokenizer) used for probabilistic accept/reject scoring.
# "Decision: ACCEPT" -> " ACCEPT" (53060); "Decision: REJECT" -> " RE" (3476) + "JECT".
_ACCEPT_TOKEN_IDS = {53060, 93997, 10024, 16156, 63681, 49999, 52742, 11330, 87247, 25503, 23890}
_REJECT_TOKEN_IDS = {3476, 762, 45427, 75725, 84282, 7602, 92060, 75035, 17030, 57409, 60476}


def load_test_data(path: str, limit: int | None = None) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit and len(samples) >= limit:
                break
    return samples


def build_inference_prompt(text: str, dataset: str, grpo: bool = False) -> str:
    if grpo:
        # rl_grpo/train_grpo.py's build_dataset() uses the raw applicant text
        # as the user turn, with every instruction carried in the system
        # message instead -- match that exactly rather than layering
        # GERMAN_V2_INSTRUCTION on top of it too.
        return text
    if dataset == "german":
        return f"{GERMAN_V2_INSTRUCTION}\n\n{text}"
    instruction = DATASET_INSTRUCTIONS[dataset].format(text=text)
    return f"[System] {SYSTEM_PROMPT}\n\n{instruction}"


def resolve_base_model(base_model_arg: str | None, adapter_path: str) -> str:
    config_path = Path(adapter_path) / "adapter_config.json"
    recorded = None
    if config_path.exists():
        try:
            recorded = json.loads(config_path.read_text()).get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError):
            recorded = None

    if base_model_arg is None:
        if recorded:
            print(f"--base-model not given; using '{recorded}' from {config_path}")
            return recorded
        print(f"--base-model not given and {config_path} has no recorded base; "
              f"falling back to default '{DEFAULT_BASE_MODEL}'")
        return DEFAULT_BASE_MODEL

    if recorded and recorded != base_model_arg:
        print(f"!!! WARNING: --base-model '{base_model_arg}' does not match "
              f"'{recorded}' recorded in {config_path}. Proceeding with the "
              f"explicitly requested base model, but this adapter's LoRA "
              f"weights were trained against a different base -- results are "
              f"likely meaningless. Drop --base-model to auto-use the "
              f"recorded one instead.")
    return base_model_arg


def load_model(base_model: str, adapter_path: str):
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import PeftModel

    print(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print(f"Loading base model {base_model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    # unsloth/Qwen3.5-9B is a multimodal wrapper (Qwen3_5ForConditionalGeneration).
    # Training targeted model.language_model.layers.*, so we must load the same
    # wrapper or the LoRA weights won't map. Fall back to CausalLM if unavailable
    # (e.g. a merged text-only base whose config rejects the image-text loader).
    def _load_causallm():
        print("Retrying with AutoModelForCausalLM...")
        return AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    try:
        base = AutoModelForImageTextToText.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        if not hasattr(base, "model") or not hasattr(base.model, "language_model"):
            print("Wrapper lacks language_model; retrying with AutoModelForCausalLM...")
            base = _load_causallm()
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            print("Image-text loader rejected the config; retrying with AutoModelForCausalLM...")
            base = _load_causallm()
        else:
            raise

    print(f"Attaching LoRA adapter from {adapter_path}...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    for w in caught:
        if "missing adapter keys" in str(w.message):
            missing = str(w.message).count("lora_A")
            print(f"\n!!! WARNING: adapter did not fully attach ({missing} LoRA weights unmapped).")
            print("!!! The model is running WITHOUT your fine-tune — results will be random.")
            print("!!! Fix: load with AutoModelForImageTextToText (multimodal wrapper) so the")
            print("!!! LoRA keys under model.language_model.layers.* match the training layout.")
            break

    return model, tokenizer


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1400,
    temperature: float = 0.1,
    greedy: bool = False,
    system_prompt: str | None = None,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=temperature if not greedy else None,
            output_scores=True,
            return_dict_in_generate=True,
        )

    response = tokenizer.decode(
        outputs.sequences[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    accept_score = _accept_score_from_scores(outputs.scores)
    return response, accept_score


def _accept_score_from_scores(scores) -> float:
    """Probability of the ACCEPT token vs REJECT tokens at the model's decision step.

    Walks the per-step logits once, tracking the step with the most total
    accept/reject token mass (the verdict position), and returns
    P(accept) / (P(accept) + P(reject)) there, in [0, 1]. Replaces the old hard
    0/1 y_scores so ROC-AUC / PR-AUC are meaningful.
    """
    accept_ids = torch.tensor(sorted(_ACCEPT_TOKEN_IDS), dtype=torch.long, device=scores[0].device)
    reject_ids = torch.tensor(sorted(_REJECT_TOKEN_IDS), dtype=torch.long, device=scores[0].device)

    best_total = 0.0
    best_score = 0.5
    for step_logits in scores:
        probs = torch.softmax(step_logits.float().squeeze(0), dim=-1)
        p_accept = probs[accept_ids].sum().item()
        p_reject = probs[reject_ids].sum().item()
        total = p_accept + p_reject
        if total > best_total:
            best_total = total
            best_score = p_accept / total if total > 1e-9 else 0.5
    return best_score


def _verdict(match: re.Match) -> int:
    word = match.group(1).lower()
    return 1 if word.startswith("accept") else 0


def _last_match(pattern: re.Pattern, text: str) -> re.Match | None:
    """Last match instead of first, over the FULL text (not just the tail).
    See the module-level comment above the pattern definitions for why."""
    m = None
    for m in pattern.finditer(text):
        pass
    return m


def extract_label(response: str, accept_score: float | None = None) -> int | None:
    """Extract binary label (1=accept, 0=reject) from the model's reasoning text.

    Checked in order of confidence: labeled verdict markers (bold / "Decision:"
    / "Conclusion:" etc.) and a bare accept/reject word are compared against
    each other by TEXT POSITION (not tier priority) -- picking "the first
    tier with any match" would mean a restatement in a lower-priority tier
    could never win even if it came later. E.g. "Decision: ACCEPT. On
    reflection I REJECT this." must resolve to reject, not accept. Falls
    through to legacy <answer> tags, then a bare good/bad word. If no textual
    verdict is found at all, falls back to the model's own implicit
    token-probability lean (accept_score) instead of silently guessing the
    majority class -- the caller decides what to do if this still returns
    None (that means even the logit lean was uninformative).
    """
    best_match = None
    for pattern in (_BOLD_VERDICT, _DECISION_LINE, _CONCLUSION_LINE, _VERDICT_WORD):
        m = _last_match(pattern, response)
        if m is not None and (best_match is None or m.end() > best_match.end()):
            best_match = m
    if best_match is not None:
        return _verdict(best_match)

    # Legacy <reasoning>/<answer> tag format.
    parsed = parse_response(response)
    if parsed is not None:
        _, answer = parsed
        return 1 if answer == "good" else 0

    m = _last_match(_GOOD_BAD_WORD, response)
    if m:
        return 1 if m.group(1).lower() == "good" else 0

    # Nothing textual to go on -- use the model's own logit lean if it has one
    # (accept_score is 0.5 only when the tracked accept/reject tokens never
    # showed meaningful probability mass anywhere in the generation).
    if accept_score is not None and accept_score != 0.5:
        return 1 if accept_score > 0.5 else 0

    return None


def run_evaluation(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from {args.test_file}...")
    samples = load_test_data(args.test_file, args.limit)
    print(f"Loaded {len(samples)} test samples.")

    if not samples:
        print("No test samples found. Exiting.")
        sys.exit(1)

    base_model = resolve_base_model(args.base_model, args.adapter_path)
    model, tokenizer = load_model(base_model, args.adapter_path)
    system_prompt = args.system_prompt or (GRPO_SYSTEM_PROMPT if args.grpo else None)

    y_true = []
    y_pred = []
    y_scores = []
    parse_failures = 0       # no textual verdict AND accept_score was uninformative
    score_fallbacks = 0      # no textual verdict, but accept_score rescued it
    results_detail = []

    print(f"\nRunning inference on {len(samples)} samples...")
    start_time = time.time()

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        text = sample["text"]
        label = sample["target"]

        prompt = build_inference_prompt(text, args.dataset, grpo=args.grpo)
        response, score = generate_prediction(
            model,
            tokenizer,
            prompt,
            temperature=args.temperature,
            greedy=args.greedy,
            max_new_tokens=args.max_new_tokens,
            system_prompt=system_prompt,
        )

        text_pred = extract_label(response)  # text-only, for honest bookkeeping
        pred = text_pred if text_pred is not None else extract_label(response, accept_score=score)
        # Only a true failure if even the logit-lean fallback had nothing to
        # go on. Track it honestly instead of silently writing a
        # majority-class guess into predicted_label, and keep the sample in
        # the metrics (dropping it would shrink the denominator and quietly
        # inflate accuracy on whatever's left -- parse failures are not a
        # random subset, they skew toward the model's worst outputs).
        parse_failure = pred is None
        if parse_failure:
            parse_failures += 1
            pred = 1  # default to majority class only as the last resort
        elif text_pred is None:
            score_fallbacks += 1

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(score)

        results_detail.append({
            "index": i,
            "text": text,
            "true_label": "good" if label == 1 else "bad",
            "predicted_label": "good" if pred == 1 else "bad",
            "parse_failure": parse_failure,
            "used_score_fallback": text_pred is None and not parse_failure,
            "accept_score": round(score, 6),
            "correct": label == pred,
            "raw_output": response,
        })

    elapsed = time.time() - start_time

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    metrics = compute_all(y_true, y_pred, y_scores)
    metrics["parse_failures"] = parse_failures
    metrics["parse_failure_rate"] = parse_failures / len(samples)
    metrics["score_fallbacks"] = score_fallbacks
    metrics["total_samples"] = len(samples)
    metrics["elapsed_seconds"] = round(elapsed, 2)
    metrics["samples_per_second"] = round(len(samples) / elapsed, 4)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset:       {args.dataset}")
    print(f"Total samples: {len(samples)}")
    print(f"Text-parsed:   {len(samples) - parse_failures - score_fallbacks}")
    print(f"Score fallback:{score_fallbacks} (no textual verdict, used token-probability lean)")
    print(f"Parse failures:{parse_failures} ({metrics['parse_failure_rate']:.1%}) "
          f"-- defaulted to majority class, still counted in metrics below")
    print(f"Time:          {elapsed:.1f}s ({metrics['samples_per_second']} samples/s)")
    print(f"{'='*60}")
    print(metrics["classification_report"])
    print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
    print(f"F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"F1 (binary):   {metrics['f1']:.4f}")
    print(f"F2 Score:      {metrics['f2']:.4f}")
    print(f"MCC:           {metrics['mcc']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"eval_{args.dataset}_{timestamp}.json"

    serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    serializable["classification_report"] = metrics["classification_report"]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nMetrics saved to {report_path}")

    detail_path = RESULTS_DIR / f"eval_{args.dataset}_{timestamp}_details.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in results_detail:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Per-sample details saved to {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="TabLens Evaluation Script")
    parser.add_argument(
        "--base-model", default=None,
        help=f"Base model path. Default: auto-detected from the adapter's own "
        f"adapter_config.json (base_model_name_or_path), falling back to "
        f"'{DEFAULT_BASE_MODEL}' if that's missing. Pass explicitly to override.",
    )
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH, help="LoRA adapter path")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["german", "loans"])
    parser.add_argument(
        "--grpo", action="store_true",
        help="Evaluate a GRPO-trained checkpoint: prompts with "
        "rl_grpo.prompts.SYSTEM_PROMPT as a system message and the raw "
        "applicant text as the user turn, matching rl_grpo/train_grpo.py's "
        "build_dataset() exactly instead of the plain-SFT prompt format.",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="Ad hoc system-prompt override (e.g. to re-test a checkpoint "
        "trained under an older/different SYSTEM_PROMPT than the current "
        "one in rl_grpo/prompts.py). Overrides --grpo's default if both are "
        "given. Use this instead of hand-editing GERMAN_V2_INSTRUCTION for a "
        "one-off need -- that constant must stay matched to the SFT data.",
    )
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Path to test JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument(
        "--greedy", action="store_true", help="Greedy decoding (deterministic, faster)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1400,
        help="Max tokens per sample. The verdict appears in the first ~100 tokens, "
        "so ~140 is enough for a fast accuracy check (small tradeoff).",
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

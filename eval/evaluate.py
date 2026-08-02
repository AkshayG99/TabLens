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

_REJECT_BOLD = re.compile(r"\*\*REJECTED?\*\*", re.IGNORECASE)
_ACCEPT_BOLD = re.compile(r"\*\*ACCEPTED?\*\*", re.IGNORECASE)
_REJECT_DECISION = re.compile(r"\bDecision\s*:?\s*\**\s*REJECT", re.IGNORECASE)
_ACCEPT_DECISION = re.compile(r"\bDecision\s*:?\s*\**\s*ACCEPT", re.IGNORECASE)
_REJECT_TO = re.compile(r"\bdecisions?\s+to\s+REJECT", re.IGNORECASE)
_ACCEPT_TO = re.compile(r"\bdecisions?\s+to\s+ACCEPT", re.IGNORECASE)
_REJECT_WORD = re.compile(r"\b(?:reject|rejects|rejected|rejecting)\b", re.IGNORECASE)
_ACCEPT_WORD = re.compile(r"\b(?:accept|accepts|accepted|accepting)\b", re.IGNORECASE)


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


def build_inference_prompt(text: str, dataset: str) -> str:
    if dataset == "german":
        return f"{GERMAN_V2_INSTRUCTION}\n\n{text}"
    instruction = DATASET_INSTRUCTIONS[dataset].format(text=text)
    return f"[System] {SYSTEM_PROMPT}\n\n{instruction}"


def load_model(base_model: str, adapter_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    print(f"Loading base model {base_model} in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

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
            print("!!! Match transformers/peft to training versions (peft 0.18.1, transformers 5.8.0).")
            break

    return model, tokenizer


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1400,
    temperature: float = 0.1,
    greedy: bool = False,
) -> str:
    messages = [{"role": "user", "content": prompt}]
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
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response


def _verdict(match: re.Match) -> int:
    word = match.group(1).lower()
    return 1 if word.startswith("accept") else 0


def extract_label(response: str) -> int | None:
    """Extract binary label (1=accept, 0=reject) from the model's reasoning text."""
    # 1. Explicit bold verdict markers (highest confidence)
    m = _REJECT_BOLD.search(response)
    if m:
        return 0
    m = _ACCEPT_BOLD.search(response)
    if m:
        return 1

    # 2. "Decision: ..." lines
    if _REJECT_DECISION.search(response):
        return 0
    if _ACCEPT_DECISION.search(response):
        return 1

    # 3. Verdict in the conclusion tail ("Conclusion/Final Decision/Verdict/Outcome: X")
    tail = response[-500:]
    m = re.search(
        r"\b(?:Conclusion|Final Decision|Verdict|Outcome)\s*:?\s*\**\s*(ACCEPTED?|REJECTED?)\b",
        tail,
        re.IGNORECASE,
    )
    if m:
        return _verdict(m)

    # 4. "decision to ACCEPT/REJECT" phrasing in the opening statement
    intro = response[:700]
    if _REJECT_TO.search(intro):
        return 0
    if _ACCEPT_TO.search(intro):
        return 1

    # 5. Decision verb in the opening statement
    if _REJECT_WORD.search(intro):
        return 0
    if _ACCEPT_WORD.search(intro):
        return 1

    # 6. Decision verb anywhere in the text
    if _REJECT_WORD.search(response):
        return 0
    if _ACCEPT_WORD.search(response):
        return 1

    # 7. Fallback to legacy good/bad language
    parsed = parse_response(response)
    if parsed is not None:
        _, answer = parsed
        return 1 if answer == "good" else 0
    lower = response.lower()
    if "good" in lower:
        return 1
    if "bad" in lower:
        return 0
    return None


def run_evaluation(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading test data from {args.test_file}...")
    samples = load_test_data(args.test_file, args.limit)
    print(f"Loaded {len(samples)} test samples.")

    if not samples:
        print("No test samples found. Exiting.")
        sys.exit(1)

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    y_true = []
    y_pred = []
    y_scores = []
    parse_failures = 0
    results_detail = []

    print(f"\nRunning inference on {len(samples)} samples...")
    start_time = time.time()

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        text = sample["text"]
        label = sample["target"]

        prompt = build_inference_prompt(text, args.dataset)
        response = generate_prediction(
            model, tokenizer, prompt, temperature=args.temperature, greedy=args.greedy
        )

        pred = extract_label(response)
        if pred is None:
            parse_failures += 1
            pred = 1  # default to majority class to avoid crashing metrics
            score = 0.0
        else:
            score = float(pred)

        y_true.append(label)
        y_pred.append(pred)
        y_scores.append(score)

        results_detail.append({
            "index": i,
            "text": text,
            "true_label": "good" if label == 1 else "bad",
            "predicted_label": "good" if pred == 1 else "bad",
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
    metrics["total_samples"] = len(samples)
    metrics["elapsed_seconds"] = round(elapsed, 2)
    metrics["samples_per_second"] = round(len(samples) / elapsed, 4)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset:       {args.dataset}")
    print(f"Samples:       {len(samples)}")
    print(f"Parse failures:{parse_failures} ({metrics['parse_failure_rate']:.1%})")
    print(f"Time:          {elapsed:.1f}s ({metrics['samples_per_second']} samples/s)")
    print(f"{'='*60}")
    print(metrics["classification_report"])
    print(f"PR-AUC:        {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:       {metrics['roc_auc']:.4f}")
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
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model path")
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH, help="LoRA adapter path")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=["german", "loans"])
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE, help="Path to test JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument(
        "--greedy", action="store_true", help="Greedy decoding (deterministic, faster)"
    )
    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

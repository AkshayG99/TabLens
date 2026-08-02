import argparse
import json
import re
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "unsloth/Qwen3.5-9B"
ADAPTER_PATH = "outputs/qwen3.5-9b-sft"

GERMAN_V2_INSTRUCTION = (
    "You are an expert credit risk analyst. Analyze the following German Credit applicant "
    "profile and provide a detailed, step-by-step Chain-of-Thought (COT) explanation of why "
    "this applicant should be accepted or rejected."
)


def load_test_samples(path: str, limit: int = 3) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def _verdict(match: re.Match) -> str:
    word = match.group(1).lower()
    return "accept" if word.startswith("accept") else "reject"


def extract_label(response: str) -> str:
    # 1. Explicit bold markers: **ACCEPT** / **ACCEPTED** / **REJECT** / **REJECTED**
    m = re.search(r"\*\*(ACCEPTED?|REJECTED?)\*\*", response, re.IGNORECASE)
    if m:
        return _verdict(m)

    # 2. "Decision: X" or "**Decision:** X" lines
    m = re.search(r"\bDecision\s*:?\s*\**\s*(ACCEPTED?|REJECTED?)\b", response, re.IGNORECASE)
    if m:
        return _verdict(m)

    # 3. Verdict in the tail (conclusion / final decision section)
    tail = response[-500:]
    m = re.search(
        r"\b(?:Conclusion|Final Decision|Verdict|Outcome)\s*:?\s*\**\s*(ACCEPTED?|REJECTED?)\b",
        tail,
        re.IGNORECASE,
    )
    if m:
        return _verdict(m)

    # 4. "decision to ACCEPT/REJECT" phrasing (the training outputs state this up front)
    intro = response[:700]
    if re.search(r"\bdecisions?\s+to\s+REJECT", intro, re.IGNORECASE):
        return "reject"
    if re.search(r"\bdecisions?\s+to\s+ACCEPT", intro, re.IGNORECASE):
        return "accept"

    # 5. Decision verb in the opening statement
    if re.search(r"\b(?:reject|rejects|rejected|rejecting)\b", intro, re.IGNORECASE):
        return "reject"
    if re.search(r"\b(?:accept|accepts|accepted|accepting)\b", intro, re.IGNORECASE):
        return "accept"

    # 6. Decision verb anywhere in the text
    if re.search(r"\b(?:reject|rejects|rejected|rejecting)\b", response, re.IGNORECASE):
        return "reject"
    if re.search(r"\b(?:accept|accepts|accepted|accepting)\b", response, re.IGNORECASE):
        return "accept"

    # 7. Legacy good/bad language
    lower = response.lower()
    if "good" in lower:
        return "accept"
    if "bad" in lower:
        return "reject"
    return "unknown"


def load_model():
    print("Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    print("Loading base model in 4-bit...")
    t1 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    t2 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        adapter_warnings = [str(w.message) for w in caught]
    model.eval()
    t3 = time.time()

    device_name = torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else "CPU"
    print(f"Device:        {model.device} ({device_name})")
    print(f"Tokenizer:     {t1 - t0:.1f}s")
    print(f"Base model:    {t2 - t1:.1f}s")
    print(f"LoRA adapter:  {t3 - t2:.1f}s")

    # Detect adapter weights that failed to attach (silent root cause of bad output)
    for w in adapter_warnings:
        if "missing adapter keys" in w:
            missing = w.count("lora_A")
            print("\n!!! WARNING: adapter did not fully attach")
            print(f"!!! {missing} LoRA weights could not be mapped to the base model.")
            print("!!! The model is running WITHOUT your fine-tune — results will be random.")
            print("!!! Fix: match transformers/peft versions to training (peft 0.18.1, transformers 5.8.0).")
            break

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="TabLens Model Inference Test")
    parser.add_argument("--test-file", default="data/processed/german_v2/test.jsonl")
    parser.add_argument("--limit", type=int, default=3, help="Number of test samples to run")
    parser.add_argument("--max-new-tokens", type=int, default=1400, help="Max generated tokens per sample")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (deterministic, faster)")
    args = parser.parse_args()

    model, tokenizer = load_model()
    samples = load_test_samples(args.test_file, args.limit)

    for i, sample in enumerate(samples, 1):
        profile = sample["text"]
        true_label = "accept" if sample["target"] == 1 else "reject"

        print(f"\n{'='*60}")
        print(f"TEST {i} (expected: {true_label})")
        print(f"{'='*60}")

        prompt = f"{GERMAN_V2_INSTRUCTION}\n\n{profile}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        t_gen = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=0.1 if not args.greedy else None,
            )
        gen_secs = time.time() - t_gen
        new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        verdict = extract_label(response)
        match = "MATCH" if verdict == true_label else "MISMATCH"
        truncated = " [TRUNCATED]" if new_tokens >= args.max_new_tokens else ""

        print(f"Verdict: {verdict}  [{match}]  (gen: {gen_secs:.1f}s, {new_tokens} tok, "
              f"{new_tokens / gen_secs:.1f} tok/s{truncated})")
        print(f"Output:\n{response}")


if __name__ == "__main__":
    main()

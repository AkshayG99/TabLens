import argparse
import json
import re

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


def extract_label(response: str) -> str:
    m = re.search(r"\*\*(ACCEPTED?|REJECTED?)\*\*", response, re.IGNORECASE)
    if m:
        return "accept" if m.group(1).lower().startswith("accept") else "reject"
    m = re.search(r"\bDecision\s*:?\s*\**\s*(ACCEPTED?|REJECTED?)\b", response, re.IGNORECASE)
    if m:
        return "accept" if m.group(1).lower().startswith("accept") else "reject"
    m = re.search(r"\b(?:reject|rejects|rejected|rejecting)\b", response, re.IGNORECASE)
    if m:
        return "reject"
    m = re.search(r"\b(?:accept|accepts|accepted|accepting)\b", response, re.IGNORECASE)
    if m:
        return "accept"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="TabLens Model Inference Test")
    parser.add_argument("--test-file", default="data/processed/german_v2/test.jsonl")
    parser.add_argument("--limit", type=int, default=3, help="Number of test samples to run")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

    print("Loading base model in 4-bit...")
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    samples = load_test_samples(args.test_file, args.limit)

    for i, sample in enumerate(samples, 1):
        profile = sample["text"]
        true_label = "accept" if sample["target"] == 1 else "reject"

        print(f"\n{'='*60}")
        print(f"TEST {i} (expected: {true_label})")
        print(f"{'='*60}")
        print(f"Profile: {profile[:150]}...")

        prompt = f"{GERMAN_V2_INSTRUCTION}\n\n{profile}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        verdict = extract_label(response)
        match = "MATCH" if verdict == true_label else "MISMATCH"
        print(f"Verdict: {verdict}  [{match}]")
        print(f"Output: {response[:600]}")


if __name__ == "__main__":
    main()

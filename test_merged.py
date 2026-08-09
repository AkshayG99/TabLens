import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from test_model import GERMAN_V2_INSTRUCTION, extract_label

MERGED_MODEL = "outputs/qwen3.5-9b-merged"


def main():
    parser = argparse.ArgumentParser(description="TabLens Merged Model Smoke Test")
    parser.add_argument("--model", default=MERGED_MODEL)
    parser.add_argument("--test-file", default="data/processed/german_v2/test.jsonl")
    parser.add_argument("--limit", type=int, default=1, help="Number of test samples to run")
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading merged model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    with open(args.test_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][: args.limit]

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
                do_sample=False,
                temperature=None,
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

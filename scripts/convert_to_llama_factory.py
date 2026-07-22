#!/usr/bin/env python3
"""Convert CoT traces to LLaMA Factory instruction/input/output format."""

import json

INPUT = "data/processed/german/train_explanations.jsonl"
OUTPUT = "data/processed/german/train_explanations_llama.jsonl"

with open(INPUT) as fin, open(OUTPUT, "w") as fout:
    for line in fin:
        row = json.loads(line)
        fout.write(json.dumps({
            "instruction": "Explain the credit decision for the following applicant.",
            "input": row["text"],
            "output": row["explanation"],
        }) + "\n")

print(f"Converted {sum(1 for _ in open(OUTPUT))} samples to {OUTPUT}")

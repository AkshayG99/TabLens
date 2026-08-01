import argparse
import asyncio
import json
import os
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from reasoning_prompt import MODEL_NAME, build_prompt, parse_response

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "reasoning"

API_KEY = os.getenv("GOOGLE_API_KEY")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))

client = genai.Client(api_key=API_KEY)

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
rate_limiter = asyncio.Semaphore(RATE_LIMIT_RPM)


def load_split(dataset: str, split: str) -> pd.DataFrame:
    return pd.read_json(PROCESSED_DIR / dataset / f"{split}.jsonl", lines=True)


def perturb_numerics(text: str, perturb_range: float = 0.05) -> str:
    import re

    def _perturb(match):
        val = float(match.group(0))
        delta = random.uniform(-perturb_range, perturb_range)
        return str(round(val * (1 + delta), 4))

    return re.sub(r"\b0?\.\d+\b", _perturb, text)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
async def call_gemma(messages: list[dict], temperature: float = 0.7) -> str:
    async with semaphore, rate_limiter:
        await asyncio.sleep(60 / RATE_LIMIT_RPM)
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=MODEL_NAME,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=2048,
                    ),
                ),
                timeout=60.0
            )
            return response.text
        except asyncio.TimeoutError:
            print("\n[DEBUG] API call timed out after 60 seconds!")
            raise


async def generate_one(text: str, dataset: str, temperature: float = 0.7) -> dict | None:
    messages = build_prompt(text, dataset)
    try:
        raw = await call_gemma(messages, temperature=temperature)
    except Exception as e:
        print(f"API error: {e}")
        return None

    if raw is None:
        print("\n[DEBUG] Generation failed: Model returned no text (possible safety filter).")
        return None

    parsed = parse_response(raw)
    if parsed is None:
        print(f"\n[DEBUG] Parsing failed. Raw model output was:\n{raw}\n")
        return None

    reasoning, answer = parsed
    return {
        "instruction": f"Analyze the following credit profile and determine if the applicant is creditworthy.",
        "input": text,
        "output": f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>",
        "label": answer,
        "dataset": dataset,
    }


async def generate_variations(
    texts: list[str], dataset: str, base_temp: float = 0.8
) -> list[dict]:
    tasks = [generate_one(text, dataset, temperature=base_temp + random.uniform(-0.1, 0.3)) for text in texts]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def process_dataset(dataset: str, target_total: int = 15000, limit: int | None = None, offset: int = 0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_reasoning.jsonl"

    train_df = load_split(dataset, "train")
    val_df = load_split(dataset, "val")
    test_df = load_split(dataset, "test")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    if offset > 0:
        all_df = all_df.iloc[offset:]
        
    if limit is not None:
        all_df = all_df.head(limit)

    base_count = len(all_df)
    variations_per_sample = max(1, (target_total - base_count) // base_count)
    print(f"{dataset}: {base_count} base samples, generating {variations_per_sample} variations each")

    records = []
    completed = 0
    
    async def process_row(i, row):
        nonlocal completed
        local_records = []
        original = row["text"]
        
        base_result = await generate_one(original, dataset, temperature=0.5)
        if base_result:
            base_result["augmentation"] = "original"
            local_records.append(base_result)

        if variations_per_sample > 0:
            var_texts = [perturb_numerics(original) for _ in range(variations_per_sample)]
            var_results = await generate_variations(var_texts, dataset)
            for r in var_results:
                r["augmentation"] = "perturbed"
            local_records.extend(var_results)
            
        completed += 1
        print(f"\r  [{dataset}] Processing samples concurrently... {completed}/{base_count} done", end="", flush=True)
        return local_records

    print(f"\n  [{dataset}] Launching {base_count} tasks concurrently (Max Concurrency: {MAX_CONCURRENCY})...")
    tasks = [process_row(i, row) for i, row in all_df.iterrows()]
    results_lists = await asyncio.gather(*tasks)
    
    for res_list in results_lists:
        records.extend(res_list)
        
    print() # newline after progress loop

    mode = "a" if offset > 0 else "w"
    with open(out_path, mode) as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"{dataset}: {len(records)} total reasoning records saved to {out_path}")
    return records


async def main(datasets: list[str], target_total: int = 15000, limit: int | None = None, offset: int = 0, dry_run: bool = False):
    if dry_run:
        for dataset in datasets:
            df = load_split(dataset, "train")
            sample = df.iloc[0]["text"]
            target = df.iloc[0]["target"]
            label = "good" if target == 1 else "bad"

            print(f"=== {dataset.upper()} DRY RUN ===")
            print(f"Input: {sample}\n")

            result = await generate_one(sample, dataset, temperature=0.5)
            if result:
                print(f"Predicted: {result['label']}")
                print(f"Actual:    {label}")
                print(f"Match:     {result['label'] == label}")
                print(f"\nFull record:\n{json.dumps(result, indent=2)}")
            else:
                print("Generation failed.")
            print()
        return

    all_records = []
    for dataset in datasets:
        records = await process_dataset(dataset, target_total=target_total, limit=limit, offset=offset)
        all_records.extend(records)

    combined_path = OUTPUT_DIR / "all_reasoning.jsonl"
    mode = "a" if offset > 0 else "w"
    with open(combined_path, mode) as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nTotal: {len(all_records)} records saved to {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CoT reasoning traces via Gemma")
    parser.add_argument("--dry-run", action="store_true", help="Test with one sample per dataset, no file output")
    parser.add_argument("--target-total", type=int, default=15000, help="Total number of samples to generate per dataset")
    parser.add_argument("--limit", type=int, help="Maximum number of base records to process")
    parser.add_argument("--offset", type=int, default=0, help="Starting index (offset) to process from")
    parser.add_argument("--datasets", nargs="+", default=["german_v2"], help="Datasets to process")
    args = parser.parse_args()
    asyncio.run(main(datasets=args.datasets, target_total=args.target_total, limit=args.limit, offset=args.offset, dry_run=args.dry_run))

# python data/generate_explanations.py --dry-run
# python data/generate_explanations.py

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from reasoning_prompt import MODEL_NAME

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed"

API_KEY = os.getenv("GOOGLE_API_KEY")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))

client = genai.Client(api_key=API_KEY) if API_KEY else None

semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
rate_limiter = asyncio.Semaphore(RATE_LIMIT_RPM)

EXPLANATION_SYSTEM_PROMPT = (
    "You are an expert credit risk analyst. You are given a loan applicant's profile and the final decision "
    "(whether the loan was accepted/good or rejected/bad). Your task is to provide a concise explanation "
    "of WHY this decision was made based on the applicant's features."
)

def build_explanation_prompt(text: str, dataset: str, target: int):
    # 1 is good/accepted, 0 is bad/rejected
    decision = "ACCEPTED (Good)" if target == 1 else "REJECTED (Bad)"
    
    if dataset == "german":
        dataset_info = "German Credit dataset. Features include age, job stability, housing, savings, credit amount, etc."
    else:
        dataset_info = "LendingClub dataset. Features include employment stability, DTI, delinquencies, credit lines, etc."
        
    instruction = (
        f"Analyze the following {dataset_info}\n\n"
        f"Profile: {text}\n\n"
        f"Final Decision: {decision}\n\n"
        "Provide a concise explanation (1 paragraph) of why this applicant was given this final decision. "
        "Focus on the most important features that support this decision."
    )
    
    return [
        types.Content(role="user", parts=[types.Part.from_text(text=f"[System] {EXPLANATION_SYSTEM_PROMPT}\n\n{instruction}")]),
    ]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
async def call_gemma(messages: list, temperature: float = 0.3) -> str:
    if not client:
        raise ValueError("Google API client is not initialized. Ensure GOOGLE_API_KEY is set in .env")
        
    async with semaphore, rate_limiter:
        await asyncio.sleep(60 / RATE_LIMIT_RPM)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=1024,
            ),
        )
        return response.text

async def generate_explanation(record: dict, dataset: str) -> dict:
    if "explanation" in record and record["explanation"]:
        return record # Skip if already generated
        
    messages = build_explanation_prompt(record["text"], dataset, record["target"])
    try:
        explanation = await call_gemma(messages, temperature=0.3)
        record["explanation"] = explanation.strip()
    except Exception as e:
        print(f"API error generating explanation: {e}")
        record["explanation"] = None
        
    return record

async def process_file(file_path: Path, dataset: str, limit: int | None = None):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    out_file_path = file_path.with_name(file_path.stem + "_explanations" + file_path.suffix)
    
    existing_explanations = {}
    if out_file_path.exists():
        with open(out_file_path, "r") as f:
            for i, line in enumerate(f):
                if line.strip():
                    rec = json.loads(line)
                    if "explanation" in rec and rec["explanation"]:
                        existing_explanations[i] = rec["explanation"]

    print(f"Processing {file_path}...")
    records = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                if i in existing_explanations:
                    rec["explanation"] = existing_explanations[i]
                records.append(rec)
                
    if limit is not None:
        records = records[:limit]
                
    total_records = len(records)
    processed_count = 0
    
    async def process_with_progress(rec):
        nonlocal processed_count
        res = await generate_explanation(rec, dataset)
        processed_count += 1
        print(f"\r  [{file_path.name}] Progress: {processed_count}/{total_records} ({(processed_count/total_records)*100:.1f}%)", end="", flush=True)
        return res
                
    tasks = [process_with_progress(rec) for rec in records]
    
    # Process with gather
    updated_records = await asyncio.gather(*tasks)
    print() # Move to next line after progress counter finishes
    
    # Write to a new file instead of overwriting
    out_file_path = file_path.with_name(file_path.stem + "_explanations" + file_path.suffix)
    with open(out_file_path, "w") as f:
        for rec in updated_records:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Completed {file_path}. Saved {len(updated_records)} records to {out_file_path.name}")

async def main(dry_run: bool = False, limit: int | None = None):
    datasets = ["german"]
    splits = ["train"]
    
    if dry_run:
        # Just test on one record
        file_path = PROCESSED_DIR / "german" / "train.jsonl"
        with open(file_path, "r") as f:
            record = json.loads(f.readline())
            
        print("=== DRY RUN ===")
        print(f"Profile: {record['text']}")
        print(f"Target: {record['target']}")
        messages = build_explanation_prompt(record["text"], "german", record["target"])
        print("\nPrompt:")
        print(messages[0].parts[0].text)
        
        explanation = await call_gemma(messages, temperature=0.3)
        print("\nGenerated Explanation:")
        print(explanation)
        return

    for dataset in datasets:
        for split in splits:
            file_path = PROCESSED_DIR / dataset / f"{split}.jsonl"
            await process_file(file_path, dataset, limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explanations for targets and add them to jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Test with one sample, no file output")
    parser.add_argument("--limit", type=int, help="Maximum number of records to process per file")
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, limit=args.limit))

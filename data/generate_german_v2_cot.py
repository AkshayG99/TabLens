import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "cot_datasets"

API_KEY = os.getenv("GOOGLE_API_KEY")
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))

client = genai.Client(api_key=API_KEY) if API_KEY else None
semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
rate_limiter = asyncio.Semaphore(RATE_LIMIT_RPM)

from reasoning_prompt import MODEL_NAME

SYSTEM_PROMPT = (
    "You are an expert credit risk analyst. "
    "Given a loan applicant's profile and the final decision (whether the loan was accepted/good or rejected/bad), "
    "provide a step-by-step Chain-of-Thought (COT) explanation of WHY this decision was made."
)

def build_cot_prompt(text: str, target: int):
    decision = "ACCEPTED (Good)" if target == 1 else "REJECTED (Bad)"
    instruction = (
        "Analyze the following German Credit applicant profile.\n"
        f"Profile: {text}\n\n"
        f"Final Decision: {decision}\n\n"
        "Provide a detailed, step-by-step Chain-of-Thought (COT) explanation of why this applicant received this decision."
    )
    return [
        types.Content(role="user", parts=[types.Part.from_text(text=f"[System] {SYSTEM_PROMPT}\n\n{instruction}")]),
    ]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
async def call_api(messages: list, temperature: float = 0.5) -> str:
    if not client:
        raise ValueError("Google API client is not initialized. Ensure GOOGLE_API_KEY is set in .env")
        
    async with semaphore, rate_limiter:
        await asyncio.sleep(60 / RATE_LIMIT_RPM)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=messages,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=4096,
            ),
        )
        if response.candidates:
            reason = response.candidates[0].finish_reason
            if reason and getattr(reason, "name", str(reason)) != "STOP":
                print(f"\n[Warning] Generation stopped early. Reason: {getattr(reason, 'name', str(reason))}")
        return response.text

async def generate_cot(record: dict) -> dict:
    messages = build_cot_prompt(record["text"], record["target"])
    
    instruction_text = (
        "You are an expert credit risk analyst. "
        "Analyze the following German Credit applicant profile and provide a detailed, "
        "step-by-step Chain-of-Thought (COT) explanation of why this applicant should be accepted or rejected."
    )
    
    try:
        explanation = await call_api(messages, temperature=0.5)
        return {
            "instruction": instruction_text,
            "input": record["text"],
            "target": record["target"],
            "output": explanation.strip() if explanation else ""
        }
    except Exception as e:
        print(f"API error generating COT: {e}")
        return {
            "instruction": instruction_text,
            "input": record["text"],
            "target": record["target"],
            "output": ""
        }

async def process_records(records: list, out_file_path: Path):
    total_records = len(records)
    processed_count = 0
    
    async def process_with_progress(rec):
        nonlocal processed_count
        res = await generate_cot(rec)
        processed_count += 1
        print(f"\r  Progress: {processed_count}/{total_records} ({(processed_count/total_records)*100:.1f}%)", end="", flush=True)
        return res
                
    tasks = [process_with_progress(rec) for rec in records]
    updated_records = await asyncio.gather(*tasks)
    print()
    
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if out_file_path.exists() else "w"
    with open(out_file_path, mode) as f:
        for rec in updated_records:
            if rec["output"]:
                f.write(json.dumps(rec) + "\n")
            
    print(f"Completed processing. Saved to {out_file_path}")


def verdict_first(output: str, target: int) -> str:
    """Prepend the decision so the verdict isn't buried at the end of the COT."""
    decision = "ACCEPT" if target == 1 else "REJECT"
    prefix = f"Decision: {decision}\n\n"
    if output.strip().startswith("Decision:"):
        return output
    return f"{prefix}{output.strip()}"

async def main(limit: int | None = None, offset: int = 0, splits: list[str] | None = None):
    dataset = "german_v2"
    splits = splits or ["train", "val", "test"]
    
    all_records = []
    for split in splits:
        file_path = PROCESSED_DIR / dataset / f"{split}.jsonl"
        if file_path.exists():
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        all_records.append(json.loads(line))
                        
    if offset > 0:
        all_records = all_records[offset:]
    if limit is not None:
        all_records = all_records[:limit]
        
    print(f"Processing {len(all_records)} records (offset: {offset}, limit: {limit})...")

    # Write a separate, verdict-first COT file PER SPLIT so val/test never
    # leak into training (the old single german_v2_cot.jsonl mixed all splits).
    for split in splits:
        out_file = OUTPUT_DIR / f"{dataset}_cot_{split}.jsonl"
        if offset == 0 and out_file.exists():
            out_file.unlink()

        split_records = []
        split_path = PROCESSED_DIR / dataset / f"{split}.jsonl"
        if split_path.exists():
            with open(split_path, "r") as f:
                split_texts = [json.loads(line)["text"] for line in f if line.strip()]
            split_records = [r for r in all_records if r["text"] in split_texts]

        if split_records:
            await process_records(split_records, out_file)

        # Rewrite the FULL file with verdict-first output (Decision first).
        # Loads every row (not just this batch) so resuming with --offset doesn't
        # drop previously-appended rows, and is idempotent via verdict_first().
        rewritten = []
        if out_file.exists():
            with open(out_file, "r") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        rec["output"] = verdict_first(rec["output"], rec["target"])
                        rewritten.append(rec)
        if rewritten:
            with open(out_file, "w") as f:
                for rec in rewritten:
                    f.write(json.dumps(rec) + "\n")
            print(f"Rewrote {len(rewritten)} verdict-first rows to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COT explanations for german_v2")
    parser.add_argument("--limit", type=int, help="Maximum number of records to process in total")
    parser.add_argument("--offset", type=int, default=0, help="Starting index (offset) to process from")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to process")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, offset=args.offset, splits=[s.strip() for s in args.splits.split(",")]))

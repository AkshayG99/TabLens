"""
Split the full COT file into train/val/test by matching each row's `input`
text against the processed split files, and restructure targets so the verdict
comes FIRST (matches how the model generates at inference and concentrates
gradient on the decision instead of burying it at the end of the reasoning).

Fixes two problems in one shot:
  1. Data leak: german_v2_cot.jsonl currently contains ALL splits (train+val+test),
     so the model was trained on the test set.
  2. Decision dilution: the teacher rationalizes a given label with the verdict at
     the END of ~800 tokens, so the accept/reject signal gets ~2 tokens of gradient.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
PROCESSED_DIR = DATA_DIR / "processed" / "german_v2"
COT_DIR = DATA_DIR / "cot_datasets"
SOURCE_COT = COT_DIR / "german_v2_cot.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split_texts(split: str) -> set[str]:
    rows = load_jsonl(PROCESSED_DIR / f"{split}.jsonl")
    return {r["text"] for r in rows}


def verdict_first(output: str, target: int) -> str:
    decision = "ACCEPT" if target == 1 else "REJECT"
    return f"Decision: {decision}\n\n{output.strip()}"


def main():
    cot_rows = load_jsonl(SOURCE_COT)
    split_texts = {s: load_split_texts(s) for s in ["train", "val", "test"]}

    buckets = {"train": [], "val": [], "test": []}
    unmatched = []
    for r in cot_rows:
        text = r["input"]
        assigned = False
        for split, texts in split_texts.items():
            if text in texts:
                buckets[split].append(r)
                assigned = True
                break
        if not assigned:
            unmatched.append(text)

    if unmatched:
        raise RuntimeError(f"{len(unmatched)} COT rows could not be matched to a split.")

    for split, rows in buckets.items():
        if not rows:
            raise RuntimeError(f"No COT rows matched to split '{split}'.")

    for split, rows in buckets.items():
        out_file = COT_DIR / f"german_v2_cot_{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                rec = {
                    "instruction": r["instruction"],
                    "input": r["input"],
                    "target": r["target"],
                    "output": verdict_first(r["output"], r["target"]),
                }
                f.write(json.dumps(rec) + "\n")
        accept = sum(1 for r in rows if r["target"] == 1)
        print(f"  {split}: {len(rows)} rows (accept={accept}, reject={len(rows)-accept}) -> {out_file.name}")

    counts = {s: len(buckets[s]) for s in buckets}
    print(f"Split complete: {counts} (total {sum(counts.values())}).")


if __name__ == "__main__":
    main()

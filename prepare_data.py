import json
import os

import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Please provide your reasoning and explanations concisely. "
    "Keep your responses short and well within the token limit to ensure they are not cut off."
)

DATA_SOURCE = "tablens/german_credit"

OUTPUT_TRAIN_FILE = "./data/train.parquet"
OUTPUT_VAL_FILE = "./data/val.parquet"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prepare_dataset(data):
    rows = []
    for item in data:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["text"]},
        ]
        rows.append(
            {
                "prompt": messages,
                "data_source": DATA_SOURCE,
                "reward_model": {"style": "rule", "ground_truth": str(item["target"])},
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_FILE), exist_ok=True)

    raw_train_data = load_jsonl("./data/processed/german_v2/train.jsonl")
    raw_val_data = load_jsonl("./data/processed/german_v2/val.jsonl")

    df_train = prepare_dataset(raw_train_data)
    df_val = prepare_dataset(raw_val_data)

    df_train.to_parquet(OUTPUT_TRAIN_FILE)
    df_val.to_parquet(OUTPUT_VAL_FILE)

    print(f"Saved {len(df_train)} rows -> {OUTPUT_TRAIN_FILE}")
    print(f"Saved {len(df_val)} rows -> {OUTPUT_VAL_FILE}")


if __name__ == "__main__":
    main()

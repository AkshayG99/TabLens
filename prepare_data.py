import os
import pandas as pd
from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================
# Replace with your specific Qwen model path/name if different
TOKENIZER_PATH = "Qwen/Qwen3.5-9B" 

# The system prompt to ensure concise responses
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Please provide your reasoning and explanations concisely. "
    "Keep your responses short and well within the token limit to ensure they are not cut off."
)

OUTPUT_TRAIN_FILE = "./data/train.parquet"
OUTPUT_VAL_FILE = "./data/val.parquet"

def prepare_dataset(data, tokenizer):
    """
    Takes a list of dictionaries with 'question' and 'answer' keys,
    applies the Qwen chat template, and returns a pandas DataFrame.
    """
    formatted_prompts = []
    
    for item in data:
        # Create the messages list including your System Prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["text"]}
        ]
        
        # Apply Qwen's chat template
        # add_generation_prompt=True adds the final <|im_start|>assistant ready for the model to generate
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Verl expects 'prompt' for the input. We also save 'expected_response'.
        # Depending on your exact Verl DisCO reward setup, you might name these differently.
        formatted_prompts.append({
            "prompt": formatted_prompt,
            "expected_response": str(item["target"])
        })
        
    return pd.DataFrame(formatted_prompts)


def main():
    print(f"Loading tokenizer: {TOKENIZER_PATH}...")
    # This will download the tokenizer if it's not cached locally
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    
    import json
    
    def load_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    print("Loading raw data...")
    raw_train_data = load_jsonl("./data/processed/german_v2/train.jsonl")
    raw_val_data = load_jsonl("./data/processed/german_v2/val.jsonl")
    
    # ------------------------------------------------------------------
    # 2. FORMAT DATASET
    # ------------------------------------------------------------------
    print("Formatting training data...")
    df_train = prepare_dataset(raw_train_data, tokenizer)
    
    print("Formatting validation data...")
    df_val = prepare_dataset(raw_val_data, tokenizer)
    
    # ------------------------------------------------------------------
    # 3. SAVE AS PARQUET
    # ------------------------------------------------------------------
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_FILE), exist_ok=True)
    
    print(f"Saving to {OUTPUT_TRAIN_FILE} and {OUTPUT_VAL_FILE}...")
    df_train.to_parquet(OUTPUT_TRAIN_FILE)
    df_val.to_parquet(OUTPUT_VAL_FILE)
    
    print("Done! Data is ready for Verl.")

if __name__ == "__main__":
    main()

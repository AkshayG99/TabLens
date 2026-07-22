import json
import os
from pathlib import Path
import sys

# Import the prompts from reasoning_prompt.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from reasoning_prompt import SYSTEM_PROMPT, GERMAN_REASONING_INSTRUCTIONS

def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "processed" / "german" / "train_explanations.jsonl"
    output_file = base_dir / "LLaMA-Factory" / "data" / "german_credit.json"
    
    if not input_file.exists():
        print(f"Input file {input_file} not found.")
        return

    output_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Format instruction with the applicant's text
            instruction = GERMAN_REASONING_INSTRUCTIONS.format(text=item['text'])
            
            # Target is 1 (good) or 0 (bad)
            answer = "good" if item['target'] == 1 else "bad"
            
            # Construct the reasoning output
            output = f"<reasoning>{item['explanation']}</reasoning>\n<answer>{answer}</answer>"
            
            # Create Alpaca format dict
            alpaca_item = {
                "instruction": instruction,
                "input": "",
                "output": output,
                "system": SYSTEM_PROMPT
            }
            output_data.append(alpaca_item)
            
    # Write to LLaMA-Factory data directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully converted {len(output_data)} records and saved to {output_file}")

if __name__ == "__main__":
    main()

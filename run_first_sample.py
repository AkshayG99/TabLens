import json
import time
import torch
for attr in ["uint16", "uint32", "uint64", "bfloat16", "float8_e4m3fn", "float8_e5m2"]:
    if not hasattr(torch, attr):
        setattr(torch, attr, getattr(torch, "int32"))
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: torch.device("cpu")
import test_model
from test_model import load_model, GERMAN_V2_INSTRUCTION, extract_label
import pandas as pd
from rich.console import Console
from rich.table import Table
import sys
import os

# Update the adapter path as requested by the user
test_model.ADAPTER_PATH = "outputs/qwen3.5-9b-sft-a6000/checkpoint-450/"

def main():
    console = Console()
    
    # 1. Load the first sample from raw tabular data to print beautifully
    raw_csv = "data/raw/german_credit_uci.csv"
    console.print(f"\n[bold blue]Loading first row from {raw_csv}...[/bold blue]")
    df = pd.read_csv(raw_csv)
    first_row = df.iloc[0]
    
    # Print the tabular data using Rich
    table = Table(title="First Sample of Tabular Data", show_header=True, header_style="bold magenta")
    table.add_column("Attribute", style="cyan", justify="right")
    table.add_column("Value", style="green")
    
    for key, value in first_row.items():
        table.add_row(str(key), str(value))
    
    console.print(table)
    
    # 2. Get the pre-processed textual representation from test.jsonl
    # The processed test split contains the formatted text that the model expects.
    test_jsonl = "data/processed/german_v2/test.jsonl"
    with open(test_jsonl, "r") as f:
        first_test_sample = json.loads(f.readline())
        
    profile = first_test_sample["text"]
    true_label = "accept" if first_test_sample["target"] == 1 else "reject"
    
    console.print("\n[bold blue]Formatted Text Profile (Input to Model):[/bold blue]")
    console.print(profile)
    console.print(f"[bold yellow]True Label (from dataset): {true_label}[/bold yellow]\n")

    # 3. Load the Model
    console.print("[bold blue]Loading Model...[/bold blue]")
    model, tokenizer = load_model()
    
    # 4. Generate Output
    console.print("\n[bold blue]Generating Prediction...[/bold blue]")
    prompt = f"{GERMAN_V2_INSTRUCTION}\n\n{profile}"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    t_gen = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1400,
            do_sample=False,
            temperature=None,
        )
    gen_secs = time.time() - t_gen
    
    # 5. Extract and print the result
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    verdict = extract_label(response)
    
    match_str = "[bold green]MATCH[/bold green]" if verdict == true_label else "[bold red]MISMATCH[/bold red]"
    
    console.print(f"\n[bold blue]Model Verdict:[/bold blue] {verdict.upper()} {match_str} (Generation Time: {gen_secs:.1f}s)")
    console.print("\n[bold blue]Detailed Model Output:[/bold blue]")
    console.print(response)

if __name__ == "__main__":
    main()

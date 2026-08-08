import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA model into a base model.")
    parser.add_argument("--base_model", type=str, default="unsloth/Qwen3.5-9B", help="Base model ID or path")
    parser.add_argument("--lora_model", type=str, default="./outputs/qwen3.5-9b-sft-a6000/checkpoint-450", help="LoRA model path")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen3.5-9b-merged", help="Output directory for merged model")
    
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu", 
        trust_remote_code=True,
    )

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    print(f"Loading PEFT adapter from: {args.lora_model}")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_model)

    print("Merging weights... This may take a while and consume memory.")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("Done! The merged model is ready to be used.")

if __name__ == "__main__":
    main()

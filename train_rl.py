import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer

# ============================================================================
# Configuration
# ============================================================================
# The HF repo ID where you uploaded your merged model
MODEL_ID = "creativelapse/qwen3.5-9b-merged" 
# The dataset containing your DPO preferences (prompt, chosen, rejected)
# For RL, you need a dataset with pairs of good/bad responses to the same prompt.
DATASET_NAME = "Anthropic/hh-rlhf" 
OUTPUT_DIR = "./rl_output"

def main():
    print(f"Loading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading merged base model from {MODEL_ID}...")
    # Loading in bfloat16 for efficiency. 
    # If VRAM is tight on Lightning, you can add `load_in_4bit=True` (requires bitsandbytes)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # MAGIC TRICK FOR RL WITH PEFT:
    # In TRL with PEFT, you don't need to load a separate "Reference Model" into memory!
    # TRL automatically disables the adapter to get the reference logits, saving massive VRAM.
    print("Configuring new LoRA adapter for RL phase...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Qwen target modules
    )
    model = get_peft_model(model, peft_config)
    
    print(f"Loading dataset {DATASET_NAME}...")
    # Make sure your dataset has 'prompt', 'chosen', and 'rejected' columns
    dataset = load_dataset(DATASET_NAME, split="train[:5%]") # Subset for testing
    
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    training_args = TrainingArguments(
        per_device_train_batch_size=2, # Adjust based on Lightning VM VRAM (e.g., A10G/A100)
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-6, # RL typically uses much lower LR than SFT
        logging_steps=10,
        output_dir=OUTPUT_DIR,
        optim="paged_adamw_32bit",
        bf16=True, # bfloat16 training
        remove_unused_columns=False,
        report_to="none" # Change to "wandb" if you want to track metrics
    )

    print("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model,
        ref_model=None, # Must be None when using PEFT in TRL
        args=training_args,
        beta=0.1, # DPO temperature/penalty
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=512,
        max_length=1024,
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print("RL Training Done!")

if __name__ == "__main__":
    main()

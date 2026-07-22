import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "unsloth/Qwen3.5-9B"
ADAPTER_PATH = "outputs/qwen3.5-9b-sft"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# Pick some test samples
test_prompts = [
    "/no_think\nThe applicant is female, aged 0.25, with job level 1.0. They live in own housing. Their saving accounts are moderate and checking account is rich. They requested a credit amount of 0.03 over 0.1 months for education.",
    "/no_think\nThe applicant is male, aged 0.09, with job level 0.33. They live in rent housing. Their saving accounts are little and checking account is little. They requested a credit amount of 0.15 over 0.5 months for business.",
    "/no_think\nThe applicant is male, aged 0.5, with job level 0.67. They live in own housing. Their saving accounts are Unknown and checking account is moderate. They requested a credit amount of 0.05 over 0.15 months for radio/TV.",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'='*60}")
    print(f"TEST {i}")
    print(f"{'='*60}")
    print(f"Input: {prompt}")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Output: {response}")

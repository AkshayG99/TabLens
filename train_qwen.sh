#run via ./train_qwen.sh




#!/bin/bash
set -e

# Activate virtual environment
source .venv/bin/activate

# Navigate to LLaMA-Factory directory
cd LLaMA-Factory

echo "Starting fine-tuning with LLaMA-Factory..."

# Run LLaMA-Factory CLI for Supervised Fine-Tuning
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset german_credit \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../models/qwen2.5-7b-german-credit \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16

echo "Fine-tuning completed. The model is saved to TabLens/models/qwen2.5-7b-german-credit."

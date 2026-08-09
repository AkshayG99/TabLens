#!/bin/bash
# Script to setup environment and run RL (DPO) training on Lightning Cloud VM

echo "Setting up environment for RL training..."

# Install required packages for training on Lightning Studio
pip install -U pip
pip install -U transformers peft trl datasets bitsandbytes accelerate huggingface_hub wandb

echo "Environment setup complete."

# Optional: Login to huggingface (useful if you want to push your model back to HF later)
# Uncomment the line below and replace with your token, or run `huggingface-cli login` manually
# huggingface-cli login --token YOUR_HF_TOKEN

echo "Starting RL (DPO) training..."

# Run the training script
python train_rl.py

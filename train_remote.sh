# source .venv/bin/activate
# colab new --gpu L4
# colab upload LLaMA-Factory/data/german_credit.json
# colab exec -f train_remote.sh
# colab download qwen2.5-7b-german-credit.tar.gz
# colab stop
#!/bin/bash
set -e

echo "=== Setting up LLaMA-Factory on Colab VM ==="

# Clone LLaMA-Factory if not already present
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

cd LLaMA-Factory

# Install LLaMA-Factory with dependencies
echo "Installing dependencies..."
pip install -e ".[torch,metrics]"

# Register the dataset dynamically using python
echo "Registering german_credit dataset in dataset_info.json..."
python3 -c "
import json
with open('data/dataset_info.json', 'r') as f:
    data = json.load(f)
data['german_credit'] = {'file_name': 'german_credit.json'}
with open('data/dataset_info.json', 'w') as f:
    json.dump(data, f, indent=2)
"

# Assuming the user ran `colab upload LLaMA-Factory/data/german_credit.json`
# The file usually ends up in the VM's home/execution directory (~/ or /content)
if [ -f "../german_credit.json" ]; then
    mv ../german_credit.json data/
    echo "Dataset moved into data directory successfully."
else
    echo "Warning: german_credit.json not found in the parent directory!"
    echo "Make sure you ran 'colab upload LLaMA-Factory/data/german_credit.json' before running this script."
fi

echo "=== Starting Fine-Tuning ==="

# Run the training
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset german_credit \
    --dataset_dir ./data \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ../qwen2.5-7b-german-credit \
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

echo "=== Packaging the Trained Model ==="
cd ..
if [ -d "qwen2.5-7b-german-credit" ]; then
    tar -czf qwen2.5-7b-german-credit.tar.gz qwen2.5-7b-german-credit
    echo "Model successfully packaged into qwen2.5-7b-german-credit.tar.gz"
    echo "You can now download it to your local machine using: colab download qwen2.5-7b-german-credit.tar.gz"
else
    echo "Error: Output directory qwen2.5-7b-german-credit not found."
fi

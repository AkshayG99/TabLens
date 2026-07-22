#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Starting SFT training with Qwen3.5-9B-Instruct (unsloth backend)..."
llamafactory-cli train configs/sft_config.yaml

echo "Training complete. Model saved to outputs/qwen3.5-9b-sft"

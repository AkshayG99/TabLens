#!/bin/bash
set -x

# ==============================================================================
# DisCO Fine-tuning Script for Qwen + LoRA
# Based on verl PR #3357: https://github.com/verl-project/verl/pull/3357
#
# MUST run from the verl repo root: disco_trainer.yaml uses a relative Hydra
# searchpath (file://verl/trainer/config) that resolves from CWD. We cd there
# automatically. Data paths are made absolute so they still resolve.
#
# NOTE (smoke test): verl lives at ~/verl, NOT $SCRIPT_DIR/verl.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Define Model and System Prompt
# Set your model path here once you have uploaded/merged the Qwen model with LoRA.
export MODEL_PATH="Qwen/Qwen2.5-0.5B"

# We define a system prompt to ensure explanations are concise and fit within token limits.
# In verl, you typically format this into your parquet dataset's "prompt" column.
export SYSTEM_PROMPT="You are a helpful AI assistant. Please provide your reasoning and explanations concisely. Keep your responses short and well within the token limit to ensure they are not cut off."

# 2. Define Dataset Paths
# Make sure your data preparation script includes the SYSTEM_PROMPT in the chat template!
TRAIN_DATA_FILE="$SCRIPT_DIR/data/train.parquet"
VAL_DATA_FILE="$SCRIPT_DIR/data/val.parquet"

# 3. DisCO algorithm parameters
loss_mode='disco'
score_func='logL'  # Options: 'logL', 'Lratio'
tau=10             # tau=10 is recommended for 'logL', tau=1 is recommended for 'Lratio'

# 4. Hardware configurations
# Adjust these based on your available hardware (e.g. 8x GPUs per node)
nnodes=1
n_gpus_per_node=1 
ppo_micro_batch_size_per_gpu=1
rollout_n=1

# 5. Launch Training from the verl repo root (required for the Hydra searchpath)
if [ -d "$HOME/verl" ]; then
    cd "$HOME/verl"
elif [ -d "$SCRIPT_DIR/verl" ]; then
    cd "$SCRIPT_DIR/verl"
else
    echo "ERROR: verl repo not found at $HOME/verl or $SCRIPT_DIR/verl" >&2
    exit 1
fi
# Uses recipe.disco.main_disco from the PR. Reward function is TabLens' own
# binary credit-verdict scorer (the recipe default is a math \boxed{} checker).
python3 -m recipe.disco.main_disco \
    algorithm.adv_estimator=disco \
    algorithm.filter_groups.enable=False \
    custom_reward_function.path=$SCRIPT_DIR/reward/credit_reward.py \
    custom_reward_function.name=credit_reward_fn \
    data.train_files=$TRAIN_DATA_FILE \
    data.val_files=$VAL_DATA_FILE \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    +actor_rollout_ref.ref.enable=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.loss_mode=$loss_mode \
    actor_rollout_ref.actor.policy_loss.score_func=$score_func \
    actor_rollout_ref.actor.policy_loss.delta=1e-4 \
    actor_rollout_ref.actor.policy_loss.beta=1e3 \
    actor_rollout_ref.actor.policy_loss.tau=$tau \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl-disco' \
    trainer.experiment_name='Qwen-LoRA-disco-logL' \
    trainer.balance_batch=False \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=1 "${@:1}" \
    trainer.resume_mode=auto

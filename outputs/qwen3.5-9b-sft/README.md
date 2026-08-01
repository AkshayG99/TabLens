---
library_name: peft
license: other
base_model: unsloth/Qwen3.5-9B
tags:
- base_model:adapter:unsloth/Qwen3.5-9B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3.5-9b-sft
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3.5-9b-sft

This model is a fine-tuned version of [unsloth/Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B) on the train_explanations dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.1
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.18.1
- Transformers 5.8.0
- Pytorch 2.6.0+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2
# AI Agent Execution Framework: TabReason Credit Risk Modeling Pipeline

## Overview
This document provides a comprehensive, step-by-step execution framework for AI agents to construct, train, and evaluate a Tabular Reasoning (TabReason) model for credit risk prediction. The pipeline spans data preprocessing, Synthetic Chain-of-Thought (CoT) generation via Google Gemma, Supervised Fine-Tuning (SFT) using LLaMA-Factory with Qwen2.5-7B-Instruct, Reinforcement Learning (RL) using the DisCO framework via `verl`, and robust evaluation.

**Agent Directives:** 
- Strict adherence to file structures, variable naming, and configuration parameters is required.
- Do not hallucinate data; follow the deterministic steps defined below.
- Treat all tabular data as natural language serialized text for LLM ingestion.

---

## Project Structure
Agents must construct the following directory tree and populate it with the corresponding files before executing training loops.

```text
TabReason/
├── data/
│   ├── german_credit_risk.csv
│   ├── loans_full_schema.csv
│   ├── preprocess.py
│   ├── splits.py
│   ├── generate_reasoning.py
│   └── reasoning_prompt.py
├── configs/
│   └── sft_config.yaml
├── scripts/
│   └── run_sft.sh
├── rl/
│   ├── rewards.py
│   ├── train_disco.py
│   └── configs/
│       └── disco_config.yaml
├── eval/
│   ├── metrics.py
│   ├── evaluate.py
│   └── interpretability.py
├── requirements.txt
└── README.md
```

---

## Execution Phases

### Phase 1: Data Ingestion & Preprocessing
**Objective:** Prepare tabular datasets for LLM ingestion by serializing them into natural language and splitting them appropriately.

**Tasks for `data/preprocess.py` and `data/splits.py`:**
1. **Load Data:** Read `german_credit_risk.csv` and `loans_full_schema.csv`.
2. **Handle Missing Values:** 
   - German Credit: Map `"NA"` strings in `Saving accounts` and `Checking account` to `"Unknown"` or impute based on standard financial practices.
3. **Normalization:** 
   - Apply Min-Max scaling or Z-score normalization to all numerical features to ensure stability in text serialization.
4. **Target Variable Definition:**
   - German Credit: `Risk` -> binary mapping (`good` = 1, `bad` = 0).
   - LendingClub: `loan_status` -> binary mapping (`Fully Paid` = 1, `Charged Off`/`Default` = 0). Drop intermediate statuses (e.g., "Current").
5. **Natural Language Serialization:** Convert each row into a structured paragraph.
   - *Format Requirement:* `The applicant has a checking account balance of [X], savings of [Y], and a requested loan amount of [Z]...`
6. **Data Splitting:** Implement stratified splitting (80% Train, 10% Val, 10% Test) ensuring minority class (defaults/bad credit) distribution is maintained across splits.

---

### Phase 2: Chain-of-Thought (CoT) Reasoning Generation
**Objective:** Generate synthetic reasoning traces to teach the model *how* to evaluate credit risk.

**Tasks for `data/reasoning_prompt.py` and `data/generate_reasoning.py`:**
1. **Prompt Template Implementation (`reasoning_prompt.py`):**
   - Instruct the teacher model (Gemma) to:
     - Analyze each feature's relevance sequentially.
     - Identify distinct risk factors and positive financial indicators.
     - Provide a step-by-step logical deduction.
     - **Strict Output Format:** Must output EXACTLY `<reasoning>[Your step-by-step logic]</reasoning><answer>[label]</answer>`.
2. **API Execution (`generate_reasoning.py`):**
   - Call Google AI Studio Gemma API.
   - **Concurrency & Resilience:** Implement asynchronous requests with exponential backoff and strict rate limiting (e.g., using `tenacity` and `asyncio.Semaphore`).
3. **Data Augmentation:**
   - Total Target: 15K - 30K examples.
   - Base: Process all 11K original samples.
   - Variations: Generate 2-3 reasoning variations per sample using high-temperature API calls.
   - Perturbations: Slightly alter numerical features (e.g., ±5% to income) to create new synthetic reasoning paths.
4. **Export:** Save the final dataset as a JSONL file strictly compatible with LLaMA-Factory `{"instruction": "...", "input": "...", "output": "..."}`.

---

### Phase 3: Supervised Fine-Tuning (SFT)
**Objective:** Fine-tune `Qwen2.5-7B-Instruct` on the generated CoT reasoning data.

**Tasks for `configs/sft_config.yaml` and `scripts/run_sft.sh`:**
1. **Model Specs:** Base model must be `Qwen/Qwen2.5-7B-Instruct`.
2. **Methodology:** Use QLoRA.
   - Quantization: 4-bit.
   - LoRA Config: `rank = 16`, `lora_alpha = 32`, target modules = all linear.
3. **Hyperparameters (`sft_config.yaml`):**
   - `cutoff_len` (Max sequence length): 1024
   - `learning_rate`: 2e-5
   - `num_train_epochs`: 3 to 5
   - `per_device_train_batch_size`: 4
   - `gradient_accumulation_steps`: 4
   - `lr_scheduler_type`: cosine
4. **Execution Script:** `run_sft.sh` should wrap the `llamafactory-cli train configs/sft_config.yaml` command.

---

### Phase 4: Reinforcement Learning (DisCO via `verl`)
**Objective:** Optimize the model's reasoning capabilities using outcome-based and format-based rewards.

**Tasks for `rl/rewards.py` and `rl/train_disco.py`:**
1. **Reward Function Implementation (`rewards.py`):**
   - Implement a multi-component reward system (Max 2.0 points per sample).
   - **Format Reward (0.5 pts):** Awarded if the output strictly contains `<reasoning>...</reasoning>` and `<answer>...</answer>` tags.
   - **Validity Reward (0.5 pts):** Awarded if the reasoning is logically coherent and parses correctly (no broken XML/HTML structures).
   - **Correctness Reward (1.0 pts):** Awarded if the final label inside `<answer>` matches the ground truth.
   - **Class Weighting:** If the sample belongs to the minority class (e.g., `bad` credit / `Charged Off`), multiply the *Correctness Reward* by `1.5x` to penalize false negatives heavily.
2. **Configuration (`disco_config.yaml`):** Set up PPO/GRPO parameters compatible with `verl`, pointing to the SFT model weights as the initialization checkpoint.

---

### Phase 5: Evaluation & Interpretability
**Objective:** Measure model performance emphasizing minority class detection, and extract feature attributions.

**Tasks for `eval/metrics.py`, `eval/evaluate.py`, and `eval/interpretability.py`:**
1. **Evaluation Metrics (`metrics.py`):**
   - **PR-AUC** (Precision-Recall Area Under Curve): Primary metric due to class imbalance.
   - **F2 Score**: Emphasizes Recall over Precision (critical in credit risk to catch defaults).
   - **MCC** (Matthews Correlation Coefficient): To evaluate the quality of binary classifications.
2. **Benchmarking:** Compare final RL-tuned model outputs against TabReason baseline statistics.
3. **Interpretability (`interpretability.py`):**
   - **LIME:** Implement Local Interpretable Model-agnostic Explanations for individual local predictions (wrap the LLM text-generation pipeline into a probability-like output function).
   - **SHAP:** Implement SHapley Additive exPlanations for global feature importance.
   - **Comparative Analysis:** Automate a script to compare the LLM's generated textual reasoning against the top features identified by SHAP/LIME.

---

## Master Execution Pipeline
To run the full end-to-end framework, agents should execute the following terminal commands in sequence from the root `TabReason/` directory:

```bash
# 1. Prepare Data
python data/preprocess.py

# 2. Generate Synthetic CoT
python data/generate_reasoning.py

# 3. Supervised Fine-Tuning
llamafactory-cli train configs/sft_config.yaml

# 4. Reinforcement Learning
python rl/train_disco.py --config rl/configs/disco_config.yaml

# 5. Evaluate and Analyze
python eval/evaluate.py
```
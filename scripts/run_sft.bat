@echo off
REM Run TabLens SFT training on Windows (uses the `tablens` conda env).
setlocal

REM Reduce CUDA memory fragmentation (avoids OOM near peak).
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set "PYTHON=C:\Users\Gurshaan\miniconda3\envs\tablens\python.exe"
set "PROJECT_DIR=%~dp0.."

if not exist "%PYTHON%" (
    echo ERROR: tablens python not found at %PYTHON%
    exit /b 1
)

cd /d "%PROJECT_DIR%"

echo Starting SFT training with Qwen3.5-9B (unsloth backend)...
echo Dataset: data/cot_datasets/german_v2_cot_train.jsonl
"%PYTHON%" -m llamafactory.cli train configs/sft_config.yaml

echo.
echo Training complete. Model saved to outputs/qwen3.5-9b-sft
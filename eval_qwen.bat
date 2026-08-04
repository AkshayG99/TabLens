@echo off
rem run via eval_qwen.bat [adapter_path] [dataset]
rem example: eval_qwen.bat outputs/qwen3.5-9b-sft-a6000/checkpoint-450 german

setlocal

if "%~1"=="" (
    set ADAPTER_PATH=outputs/qwen3.5-9b-sft-a6000/checkpoint-450
) else (
    set ADAPTER_PATH=%~1
)

if "%~2"=="" (
    set DATASET=german
) else (
    set DATASET=%~2
)

echo Evaluating adapter: %ADAPTER_PATH% (dataset: %DATASET%)

python eval/evaluate.py ^
    --dataset %DATASET% ^
    --adapter-path %ADAPTER_PATH%

endlocal

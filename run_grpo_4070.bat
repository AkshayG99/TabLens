@echo off
rem GRPO RL fine-tuning - RTX 4070 profile (4-bit QLoRA), native Windows.
rem Usage:   run_grpo_4070.bat                       full run
rem          run_grpo_4070.bat --max-steps 3 ...     smoke test first!
rem Extra args are passed straight through to rl_grpo/train_grpo.py.

setlocal

set MODEL_PATH=outputs\qwen3.5-9b-merged
if not exist "%MODEL_PATH%\" set MODEL_PATH=creativelapse/qwen3.5-9b-merged

if not exist logs mkdir logs
set LOGFILE=logs\grpo_4070_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

echo Model: %MODEL_PATH%
echo Log:   %LOGFILE%

python rl_grpo/train_grpo.py ^
    --profile 4070 ^
    --model "%MODEL_PATH%" ^
    %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOGFILE%'"

endlocal

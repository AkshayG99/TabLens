@echo off
rem GRPO RL fine-tuning - A6000 wrapper (Windows), mirrors run_grpo_a6000.sh.
rem Model defaults to creativelapse/qwen3.5-9b-merged. Use --profile to force
rem a VRAM tier; train_grpo.py auto-detects otherwise. Extra args pass through.
rem   run_grpo_a6000.bat --num-generations 4 --per-device-batch 4 --grad-accum 8 --beta 0.02

setlocal

set MODEL_PATH=creativelapse/qwen3.5-9b-merged

if not exist logs mkdir logs
set LOGFILE=logs\grpo_a6000_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

echo Model: %MODEL_PATH%
echo Log:   %LOGFILE%

python rl_grpo/train_grpo.py ^
    --model "%MODEL_PATH%" ^
    %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOGFILE%'"

endlocal

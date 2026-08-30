@echo off
rem DisCO GRPO fine-tuning - Windows, mirrors run_disco.sh.
rem Model defaults to creativelapse/qwen3.5-9b-merged; set MODEL_PATH to a
rem local folder (outputs\...) to use it. Extra args pass through to the script.
rem   run_disco.bat --num-generations 4 --per-device-batch 4 --grad-accum 8

setlocal

set HF_DEFAULT=creativelapse/qwen3.5-9b-merged
set MODEL_PATH=%MODEL_PATH%
if not defined MODEL_PATH set MODEL_PATH=%HF_DEFAULT%

set TOKENIZERS_PARALLELISM=false
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set PATH_PREFIX=%MODEL_PATH:~0,8%
if /i "%PATH_PREFIX%"=="outputs\" if not exist "%MODEL_PATH%\" (
    echo [run_disco] WARNING: '%MODEL_PATH%' is not a folder on this machine.
    echo [run_disco]          Falling back to HF hub: %HF_DEFAULT%
    set MODEL_PATH=%HF_DEFAULT%
)

if not exist logs mkdir logs
set LOGFILE=logs\disco_run_%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOGFILE=%LOGFILE: =0%

echo [run_disco] model : %MODEL_PATH%
echo [run_disco] log   : %LOGFILE%

python rl_grpo/train_grpo.py --model "%MODEL_PATH%" --disco %* 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath '%LOGFILE%'"

endlocal

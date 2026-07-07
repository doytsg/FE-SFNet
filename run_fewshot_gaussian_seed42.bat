@echo off
setlocal enabledelayedexpansion

REM Few-shot Gaussian-noise training experiments.
REM Shots per class: 10, 20, 50, 100, 200
REM Train SNRs: -2, -4, -6, -8, -10, -12 dB
REM Seed: 42
REM Test SNR list uses the default from train_common.py.
REM Excluded models: mslk, sds_dsfb_no_adaptive_shrink.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"

set "SHOTS=10 20 50 100 200"
set "SNRS=-2 -4 -6 -8 -10 -12"
set "SEED=42"
set "COMMON_ARGS=--train_noise --val_noise --snr_per_sample --test_noise --epochs 100 --seed %SEED%"

echo ============================================
echo Few-shot Gaussian-noise experiments
echo Shots per class: %SHOTS%
echo Train SNRs: %SNRS%
echo Seed: %SEED%
echo Test SNR list: default from train_common.py
echo Logs: logs
echo CSV copies: results
echo ============================================

for %%N in (%SHOTS%) do (
    for %%S in (%SNRS%) do (
        set "SHOT=%%N"
        set "SNR=%%S"
        set "SNR_FLOAT=%%S.0"

        echo.
        echo ============================================
        echo shot %%N per class, train SNR %%S dB: running 9 tasks
        echo ============================================

        call :run_task "liconvformer" "train_liconvformer.py" "liconvformer" ""
        call :run_task "drsn_cw" "DRSN-CW.py" "drsn_cw" ""
        call :run_task "gtfenet" "GTFENET.py" "gtfenet" ""
        call :run_task "almformer" "ALMformer.py" "almformer" ""
        call :run_task "convformer_nse" "train_convformer_nse.py" "convformer_nse" ""
        call :run_task "tslanet" "TSLANet_classification.py" "tslanet" ""
        call :run_task "wdcnn" "WDCNN.py" "wdcnn" ""
        call :run_task "sds_dsfb_adaptive_shrink" "train_model.py sds_dsfb" "sds_dsfb_transformer" "--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head"
        call :run_task "cnn_transformer" "train_cnn_transformer.py" "cnn_transformer" ""

        echo shot %%N per class, train SNR %%S dB completed.
    )
)

echo.
echo ============================================
echo All experiments completed. Total tasks: 270
echo Logs saved in logs directory.
echo CSV copies saved in results directory.
echo ============================================
REM pause
exit /b 0

:run_task
set "LABEL=%~1"
set "SCRIPT=%~2"
set "CSV_KEY=%~3"
set "EXTRA_ARGS=%~4"
set "LOG_PATH=logs\!LABEL!_shot!SHOT!_snr!SNR!_seed%SEED%_%timestamp%.log"
set "SRC_CSV=results\training_history_!CSV_KEY!_snr!SNR_FLOAT!_!SNR_FLOAT!.csv"
set "DST_CSV=results\training_history_!LABEL!_shot!SHOT!_snr!SNR!_seed%SEED%_%timestamp%.csv"

echo   [!LABEL!] shot=!SHOT! train_snr=!SNR!dB seed=%SEED%
if exist "!SRC_CSV!" del /Q "!SRC_CSV!"
python !SCRIPT! !COMMON_ARGS! --train_samples_per_class !SHOT! --train_snr_min !SNR! --train_snr_max !SNR! !EXTRA_ARGS! > "!LOG_PATH!" 2>&1
if errorlevel 1 (
    echo       FAILED. See: !LOG_PATH!
) else (
    echo       Done. Log: !LOG_PATH!
)

if exist "!SRC_CSV!" (
    copy /Y "!SRC_CSV!" "!DST_CSV!" >nul
    echo       CSV: !DST_CSV!
) else (
    echo       WARNING: CSV not found: !SRC_CSV!
)
exit /b 0
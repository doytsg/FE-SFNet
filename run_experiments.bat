@echo off
setlocal enabledelayedexpansion

REM Multi-seed noisy-train experiments.
REM SNRs: -12, -10, -8, -6, -4, -2, 0
REM Seeds: 1, 2, 3, 4, 5
REM Each log/CSV name includes model/config, SNR, seed, and timestamp.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"

set "SNRS=-12 -10 -8 -6 -4 -2 0"
set "SEEDS=1 2 3 4 5"
set "COMMON_ARGS=--train_noise --val_noise --snr_per_sample --test_noise --epochs 100"

echo ============================================
echo Running noisy-train multi-seed experiments
echo SNRs: %SNRS%
echo Seeds: %SEEDS%
echo Logs: logs
echo CSV copies: results
echo ============================================

for %%S in (%SNRS%) do (
    for %%D in (%SEEDS%) do (
        set "SNR=%%S"
        set "SEED=%%D"
        set "SNR_FLOAT=%%S.0"

        echo.
        echo ============================================
        echo SNR %%S dB, seed %%D: running 11 tasks
        echo ============================================

        call :run_task "liconvformer" "train_liconvformer.py" "liconvformer" ""
        call :run_task "drsn_cw" "DRSN-CW.py" "drsn_cw" ""
        call :run_task "gtfenet" "GTFENET.py" "gtfenet" ""
        call :run_task "almformer" "ALMformer.py" "almformer" ""
        call :run_task "convformer_nse" "train_convformer_nse.py" "convformer_nse" ""
        call :run_task "tslanet" "TSLANet_classification.py" "tslanet" ""
        call :run_task "wdcnn" "WDCNN.py" "wdcnn" ""
        call :run_task "mslk" "train_mgstl_transformer_1_12.py" "mslk" "--use_asb --use_icb"

        call :run_task "sds_dsfb_no_adaptive_shrink" "train_model.py sds_dsfb" "sds_dsfb_transformer" "--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head --no_adaptive_shrink"
        call :run_task "sds_dsfb_adaptive_shrink" "train_model.py sds_dsfb" "sds_dsfb_transformer" "--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head"
        call :run_task "cnn_transformer" "train_cnn_transformer.py" "cnn_transformer" ""

        echo SNR %%S dB, seed %%D completed.
    )
)

echo.
echo ============================================
echo All experiments completed. Total tasks: 385
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
set "LOG_PATH=logs\!LABEL!_snr!SNR!_seed!SEED!_%timestamp%.log"
set "SRC_CSV=results\training_history_!CSV_KEY!_snr!SNR_FLOAT!_!SNR_FLOAT!.csv"
set "DST_CSV=results\training_history_!LABEL!_snr!SNR!_seed!SEED!_%timestamp%.csv"

echo   [!LABEL!] SNR=!SNR!dB seed=!SEED!
if exist "!SRC_CSV!" del /Q "!SRC_CSV!"
python !SCRIPT! !COMMON_ARGS! --train_snr_min !SNR! --train_snr_max !SNR! --seed !SEED! !EXTRA_ARGS! > "!LOG_PATH!" 2>&1
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
@echo off
setlocal enabledelayedexpansion

REM Clean-train / clean-val experiments with noisy SNR-list testing.
REM Seeds: 1, 2, 3, 4, 5
REM Models: original baselines + two SDS-DSFB configs + CNN Transformer.
REM No train_noise or val_noise is used here.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"

set "SEEDS=1 2 3 4 5"
set "MODE_TAG=clean_train_snrlist_test"
set "CSV_TRAIN_TAG=-2_10"
set "SNR_LIST=-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10"
set "COMMON_ARGS=--test_noise --snr_list=%SNR_LIST% --epochs 100"

echo ============================================
echo Clean train/val, SNR-list test experiments
echo Seeds: %SEEDS%
echo SNR list: %SNR_LIST%
echo Logs: logs
echo CSV copies: results
echo ============================================

for %%D in (%SEEDS%) do (
    set "SEED=%%D"

    echo.
    echo ============================================
    echo Seed %%D: running 11 tasks
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

    echo Seed %%D completed.
)

echo.
echo ============================================
echo All experiments completed. Total tasks: 55
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
set "LOG_PATH=logs\!LABEL!_%MODE_TAG%_seed!SEED!_%timestamp%.log"
set "SRC_CSV=results\training_history_!CSV_KEY!_snr%CSV_TRAIN_TAG%.csv"
set "DST_CSV=results\training_history_!LABEL!_%MODE_TAG%_seed!SEED!_%timestamp%.csv"

echo   [!LABEL!] clean train/val, SNR-list test, seed=!SEED!
if exist "!SRC_CSV!" del /Q "!SRC_CSV!"
python !SCRIPT! %COMMON_ARGS% --seed !SEED! !EXTRA_ARGS! > "!LOG_PATH!" 2>&1
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
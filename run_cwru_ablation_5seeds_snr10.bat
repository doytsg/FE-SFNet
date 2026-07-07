@echo off
setlocal enabledelayedexpansion

REM CWRU Haar wavelet design comparison, train at Gaussian SNR=10 dB.
REM Only two variants:
REM   haar_legacy : original two-stage design (AdaptiveShrinkage MLP + sigmoid gate)
REM   haar_soft   : refactored single-step noise-adaptive soft-threshold (current default)
REM Everything else identical (SK fusion on, gap cross-scale coupling, paper recipe).
REM Seeds: 1..5. Total tasks: 2 variants x 5 seeds = 10.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%I"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "SEEDS=1 2 3 4 5"
set "TRAIN_NOISE=gaussian"
set "TRAIN_SNR=-10"
set "TEST_NOISE_TYPES=gaussian"
set "TEST_SNRS=-10"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=cwru --data_dir=data --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --snr_list=%TEST_SNRS% --epochs=%EPOCHS% --results_dir=%RESULT_DIR% --noise_type=%TRAIN_NOISE% --train_snr_min=%TRAIN_SNR% --train_snr_max=%TRAIN_SNR%"
set "BASE_SDSFNET=--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head"

echo ============================================
echo CWRU Haar wavelet design comparison
echo Train/Test noise: %TRAIN_NOISE% %TRAIN_SNR% dB
echo Variants: haar_legacy vs haar_soft
echo Seeds: %SEEDS%
echo Total tasks: 10
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================

for %%D in (%SEEDS%) do (
    set "SEED=%%D"

    echo.
    echo --------------------------------------------
    echo Seed=%%D: running haar_legacy and haar_soft
    echo --------------------------------------------

    call :run_variant "haar_legacy" "--haar_design legacy"
    if errorlevel 1 exit /b 1

    call :run_variant "haar_soft"   "--haar_design soft_threshold"
    if errorlevel 1 exit /b 1

    echo Seed=%%D completed.
)

echo.
echo ============================================
echo CWRU Haar design comparison completed.
echo Logs saved in %LOG_DIR%.
echo Result files saved in %RESULT_DIR%.
echo ============================================
exit /b 0

:run_variant
set "VARIANT=%~1"
set "OVERRIDES=%~2"
set "RUN_NAME=cwru_sds_dsfb_%VARIANT%_train-gaussian_snr%TRAIN_SNR%db_seed!SEED!_%timestamp%"
set "LOG_PATH=%LOG_DIR%\!RUN_NAME!.log"

echo   [!VARIANT!] seed=!SEED!
python train_model.py sds_dsfb %COMMON_ARGS% --seed=!SEED! --run_name=!RUN_NAME! %BASE_SDSFNET% %OVERRIDES% > "!LOG_PATH!" 2>&1
if errorlevel 1 (
    echo       FAILED. See: !LOG_PATH!
    exit /b 1
) else (
    echo       Done. Log: !LOG_PATH!
    echo       Results prefix: %RESULT_DIR%\!RUN_NAME!
)
exit /b 0

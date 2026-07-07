@echo off
setlocal enabledelayedexpansion

REM PU-Mix remaining SDSFNet ablations at Gaussian SNR=-4 dB.
REM A0 and A2 were already run by run_pu_a0_a2_10seeds_snr-4.bat.
REM This script only runs:
REM   A1 w/o CS-SDS Front-End          (--no_sds_frontend)
REM   A6 MH-DSFB -> MHSA               (--use_mhsa)
REM   A7 single-head DSFB              (--dsfb_num_heads 1)
REM   A9 Li-FFN -> ConvSwiGLU FFN      (--ffn_type swiglu)
REM All variants disable selective branch fusion (--no_sk_fusion).
REM Seeds: 1..10. Total tasks: 4 variants x 10 seeds = 40.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%I"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "SEEDS=1 2 3 4 5 6 7 8 9 10"
set "TRAIN_NOISE=gaussian"
set "TRAIN_SNR=-4"
set "TEST_NOISE_TYPES=gaussian"
set "TEST_SNRS=-4"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=pu --data_dir=PU_extracted --pu_condition=N15_M01_F10,N09_M07_F10,N15_M07_F04 --pu_measurement_start=1 --pu_measurement_end=3 --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --snr_list=%TEST_SNRS% --epochs=%EPOCHS% --results_dir=%RESULT_DIR% --noise_type=%TRAIN_NOISE% --train_snr_min=%TRAIN_SNR% --train_snr_max=%TRAIN_SNR%"
set "BASE_SDSFNET=--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head --no_sk_fusion"

echo ============================================
echo PU-Mix remaining ablations, Gaussian SNR=%TRAIN_SNR% dB
echo Variants: A1 A6 A7 A9
echo Seeds: %SEEDS%
echo Total tasks: 40
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================

for %%D in (%SEEDS%) do (
    set "SEED=%%D"

    echo.
    echo --------------------------------------------
    echo Seed=%%D: running A1, A6, A7, A9
    echo --------------------------------------------

    call :run_variant "a1_no_sds"      "--no_sds_frontend"
    if errorlevel 1 exit /b 1

    call :run_variant "a6_mhsa"        "--use_mhsa"
    if errorlevel 1 exit /b 1

    call :run_variant "a7_single_head" "--dsfb_num_heads 1"
    if errorlevel 1 exit /b 1

    call :run_variant "a9_swiglu_ffn"  "--ffn_type swiglu"
    if errorlevel 1 exit /b 1

    echo Seed=%%D completed.
)

echo.
echo ============================================
echo Remaining 10-seed ablations completed.
echo Logs saved in %LOG_DIR%.
echo Result files saved in %RESULT_DIR%.
echo ============================================
exit /b 0

:run_variant
set "VARIANT=%~1"
set "OVERRIDES=%~2"
set "RUN_NAME=pu_mix_sds_dsfb_%VARIANT%_train-gaussian_snr%TRAIN_SNR%db_seed!SEED!_%timestamp%"
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

@echo off
setlocal enabledelayedexpansion

REM Experiment 2: PU-LOCO leave-one-variable-out generalization (10-class).
REM Train condition: N15_M07_F10 (1500 rpm, 0.7 Nm, 1000 N) for all tasks.
REM Each test condition differs from train in exactly one variable.
REM T1: train N15_M07_F10 -> test N09_M07_F10  (speed change: 1500 -> 900 rpm)
REM T2: train N15_M07_F10 -> test N15_M01_F10  (torque change: 0.7 -> 0.1 Nm)
REM T3: train N15_M07_F10 -> test N15_M07_F04  (radial force change: 1000 -> 400 N)
REM Bearings: 10-class PU-BS-10 (K001 / KA04 KA15 KA16 KA22 KA30 / KI16 KI17 KI18 KI21).
REM Measurements: train 1..3, test 1..2 (test smaller than train).
REM Train:val = 14:3 (no test split from train domain).
REM Window 2048, stride 2048 (no overlap), channel vibration_1.
REM Train/validation noise: gaussian, SNR = -6 dB.
REM Test noise: clean (auto) + gaussian SNR = -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 dB.
REM Models: 9 (excluding mslk and sds_dsfb_no_adaptive_shrink).
REM Seeds: 1, 2, 3, 4, 5.
REM Total tasks: 3 x 5 x 9 = 135.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "SEEDS=1 2 3 4 5"
set "TRAIN_NOISE=gaussian"
set "TRAIN_SNR=-6"
set "TEST_NOISE_TYPES=gaussian"
set "TEST_SNRS=-8,-6,-4,-2,0,2,4,6,8,10"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=pu --data_dir=PU_extracted --pu_measurement_start=1 --pu_measurement_end=3 --pu_test_measurement_start=1 --pu_test_measurement_end=2 --window_size=2048 --stride=2048 --num_classes=10"
set "SPLIT_ARGS=--train_ratio=0.8235 --val_ratio=0.1765 --test_ratio=0"
set "COMMON_ARGS=%DATASET_ARGS% %SPLIT_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --snr_list=%TEST_SNRS% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"

echo ============================================
echo Experiment 2: PU-LOCO leave-one-condition-out
echo Tasks: T1 (speed), T2 (torque), T3 (radial force)
echo Train/validation noise: %TRAIN_NOISE% SNR=%TRAIN_SNR% dB
echo Test SNRs: clean + %TEST_SNRS% dB
echo Train:Val = 14:3, train measurements 1..3, test measurements 1..2
echo Seeds: %SEEDS%
echo Total tasks: 135
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================

set "TRAIN_COND=N15_M07_F10"
call :run_task_set "t1" "%TRAIN_COND%" "N09_M07_F10"
call :run_task_set "t2" "%TRAIN_COND%" "N15_M01_F10"
call :run_task_set "t3" "%TRAIN_COND%" "N15_M07_F04"

echo.
echo ============================================
echo All experiments completed. Total tasks: 135
echo Logs saved in %LOG_DIR%.
echo Result files saved in %RESULT_DIR%.
echo ============================================
REM pause
exit /b 0

:run_task_set
set "TASK_TAG=%~1"
set "TRAIN_CONDS=%~2"
set "TEST_COND=%~3"

echo.
echo ============================================
echo Task %TASK_TAG%: train=%TRAIN_CONDS%  test=%TEST_COND%
echo ============================================

for %%D in (%SEEDS%) do (
    set "SEED=%%D"

    echo.
    echo --------------------------------------------
    echo Task %TASK_TAG%, seed=%%D: running 9 models
    echo --------------------------------------------

    call :run_task "liconvformer" "liconvformer" ""
    call :run_task "drsn_cw" "drsn_cw" ""
    call :run_task "gtfenet" "gtfenet" ""
    call :run_task "almformer" "almformer" ""
    call :run_task "convformer_nse" "convformer_nse" ""
    call :run_task "tslanet" "tslanet" ""
    call :run_task "wdcnn" "wdcnn" ""
    call :run_task "sds_dsfb_adaptive_shrink" "sds_dsfb" "--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head"
    call :run_task "cnn_transformer" "cnn_transformer" ""

    echo Task %TASK_TAG%, seed=%%D completed.
)
exit /b 0

:run_task
set "LABEL=%~1"
set "MODEL_CMD=%~2"
set "EXTRA_ARGS=%~3"
set "RUN_NAME=pu_loco_%TASK_TAG%_!LABEL!_seed!SEED!_%timestamp%"
set "LOG_PATH=%LOG_DIR%\!RUN_NAME!.log"

echo   [!LABEL!] %TASK_TAG% seed=!SEED!
python train_model.py !MODEL_CMD! %COMMON_ARGS% --pu_condition=%TRAIN_CONDS% --pu_test_condition=%TEST_COND% --noise_type=%TRAIN_NOISE% --train_snr_min=%TRAIN_SNR% --train_snr_max=%TRAIN_SNR% --seed=!SEED! --run_name=!RUN_NAME! !EXTRA_ARGS! > "!LOG_PATH!" 2>&1
if errorlevel 1 (
    echo       FAILED. See: !LOG_PATH!
) else (
    echo       Done. Log: !LOG_PATH!
    echo       Results prefix: %RESULT_DIR%\!RUN_NAME!
)
exit /b 0

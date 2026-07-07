@echo off
setlocal enabledelayedexpansion

REM Experiment 2: PU Artificial-vs-Real cross-domain comparison.
REM Dataset: PU-A2R 3-class (Healthy / Outer race / Inner race).
REM Conditions: N15_M01_F10 + N09_M07_F10 + N15_M07_F04, measurements 1..2.
REM Bearings per fault class capped at 3 (Healthy stays K001 only).
REM Window: 2048, stride: 2048 (no overlap), channel=vibration_1.
REM Models: 9 models, excluding mslk and sds_dsfb_no_adaptive_shrink.
REM Directions: A->A, R->R, A->R, R->A.
REM Seeds: 1, 2, 3, 4, 5.
REM Training: clean (no train_noise), 100 epochs.
REM Test: clean (auto) + gaussian noise from -12 to 10 dB.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "SEEDS=1 2 3 4 5"
set "TEST_NOISE_TYPES=gaussian"
set "TEST_SNRS=-12,-10,-8,-6,-4,-2,0,2,4,6,8,10"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=pu_a2r --data_dir=PU_extracted --pu_condition=N15_M01_F10,N09_M07_F10,N15_M07_F04 --pu_measurement_start=1 --pu_measurement_end=2 --pu_max_bearings_per_class=3 --window_size=2048 --stride=2048 --num_classes=3"
set "COMMON_ARGS=%DATASET_ARGS% --test_noise --test_noise_types=%TEST_NOISE_TYPES% --snr_list=%TEST_SNRS% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"

echo ============================================
echo Experiment 2: PU Artificial-vs-Real cross-domain
echo Directions: A-^>A, R-^>R, A-^>R, R-^>A
echo Seeds: %SEEDS%
echo Test noise types: %TEST_NOISE_TYPES%
echo Test SNRs: %TEST_SNRS%
echo Total tasks: 180
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================

call :run_direction "a2a" "artificial" "artificial"
call :run_direction "r2r" "real"       "real"
call :run_direction "a2r" "artificial" "real"
call :run_direction "r2a" "real"       "artificial"

echo.
echo ============================================
echo All experiments completed. Total tasks: 180
echo Logs saved in %LOG_DIR%.
echo Result files saved in %RESULT_DIR%.
echo ============================================
REM pause
exit /b 0

:run_direction
set "DIR_TAG=%~1"
set "TRAIN_DOMAIN=%~2"
set "TEST_DOMAIN=%~3"

echo.
echo ============================================
echo Direction %DIR_TAG%: train_domain=%TRAIN_DOMAIN%, test_domain=%TEST_DOMAIN%
echo ============================================

for %%D in (%SEEDS%) do (
    set "SEED=%%D"

    echo.
    echo --------------------------------------------
    echo Direction %DIR_TAG%, seed=%%D: running 9 models
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

    echo Direction %DIR_TAG%, seed=%%D completed.
)
exit /b 0

:run_task
set "LABEL=%~1"
set "MODEL_CMD=%~2"
set "EXTRA_ARGS=%~3"
set "RUN_NAME=pu_a2r_%DIR_TAG%_!LABEL!_seed!SEED!_%timestamp%"
set "LOG_PATH=%LOG_DIR%\!RUN_NAME!.log"

echo   [!LABEL!] %DIR_TAG% seed=!SEED!
python train_model.py !MODEL_CMD! %COMMON_ARGS% --pu_train_domain=%TRAIN_DOMAIN% --pu_test_domain=%TEST_DOMAIN% --seed=!SEED! --run_name=!RUN_NAME! !EXTRA_ARGS! > "!LOG_PATH!" 2>&1
if errorlevel 1 (
    echo       FAILED. See: !LOG_PATH!
) else (
    echo       Done. Log: !LOG_PATH!
    echo       Results prefix: %RESULT_DIR%\!RUN_NAME!
)
exit /b 0

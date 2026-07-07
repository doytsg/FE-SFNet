@echo off
setlocal enabledelayedexpansion

REM Experiment 1: PU-Mix strong-noise comparison.
REM PU-Mix = N15_M01_F10 + N09_M07_F10 + N15_M07_F04, measurements 1..3.
REM Window: 2048, stride: 2048 (no overlap).
REM Models: 9 models, excluding mslk and sds_dsfb_no_adaptive_shrink.
REM Few-shot train samples per class: 50, 100.
REM Training noise type: gaussian.
REM Train SNRs: -12, -8, -4, 0 dB.
REM Seeds: 1, 2, 3, 4, 5.
REM Test noise types: gaussian, laplace, uniform, impulse, mixed.
REM Test SNRs: -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 dB.

if not exist "logs" mkdir logs
if not exist "results" mkdir results

set "timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "TRAIN_NOISE=gaussian"
set "SHOTS=100 200"
set "SNRS=-12 -8 -4 0"
set "SEEDS=1 2 3 4 5"
set "TEST_NOISE_TYPES=gaussian,laplace,uniform,impulse,mixed"
set "TEST_SNRS=-12,-10,-8,-6,-4,-2,0,2,4,6,8,10"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=pu --data_dir=PU_extracted --pu_condition=N15_M01_F10,N09_M07_F10,N15_M07_F04 --pu_measurement_start=1 --pu_measurement_end=3 --window_size=2048 --stride=2048"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --snr_list=%TEST_SNRS% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"

echo ============================================
echo Experiment 1: PU-Mix strong-noise comparison
echo PU-Mix: N15_M01_F10 + N09_M07_F10 + N15_M07_F04
echo Few-shot train samples per class: %SHOTS%
echo Train noise type: %TRAIN_NOISE%
echo Train SNRs: %SNRS%
echo Seeds: %SEEDS%
echo Test noise types: %TEST_NOISE_TYPES%
echo Test SNRs: %TEST_SNRS%
echo Total tasks: 360
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================

for %%N in (%SHOTS%) do (
    for %%S in (%SNRS%) do (
        for %%D in (%SEEDS%) do (
            set "SHOT=%%N"
            set "SNR=%%S"
            set "SEED=%%D"

            echo.
            echo ============================================
            echo PU-Mix shot=%%N/class gaussian train_snr=%%S dB, seed=%%D: running 9 models
            echo ============================================

            call :run_task "liconvformer" "liconvformer" ""
            call :run_task "drsn_cw" "drsn_cw" ""
            call :run_task "gtfenet" "gtfenet" ""
            call :run_task "almformer" "almformer" ""
            call :run_task "convformer_nse" "convformer_nse" ""
            call :run_task "tslanet" "tslanet" ""
            call :run_task "wdcnn" "wdcnn" ""
            call :run_task "sds_dsfb_adaptive_shrink" "sds_dsfb" "--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head"
            call :run_task "cnn_transformer" "cnn_transformer" ""

            echo PU-Mix shot=%%N/class gaussian train_snr=%%S dB, seed=%%D completed.
        )
    )
)

echo.
echo ============================================
echo All experiments completed. Total tasks: 360
echo Logs saved in %LOG_DIR%.
echo Result files saved in %RESULT_DIR%.
echo ============================================
REM pause
exit /b 0

:run_task
set "LABEL=%~1"
set "MODEL_CMD=%~2"
set "EXTRA_ARGS=%~3"
set "RUN_NAME=pu_mix_!LABEL!_shot!SHOT!_train-gaussian_snr!SNR!db_seed!SEED!_%timestamp%"
set "LOG_PATH=%LOG_DIR%\!RUN_NAME!.log"

echo   [!LABEL!] PU-Mix shot=!SHOT!/class train_noise=gaussian train_snr=!SNR!dB seed=!SEED!
python train_model.py !MODEL_CMD! !COMMON_ARGS! --train_samples_per_class !SHOT! --noise_type=!TRAIN_NOISE! --train_snr_min=!SNR! --train_snr_max=!SNR! --seed=!SEED! --run_name=!RUN_NAME! !EXTRA_ARGS! > "!LOG_PATH!" 2>&1
if errorlevel 1 (
    echo       FAILED. See: !LOG_PATH!
) else (
    echo       Done. Log: !LOG_PATH!
    echo       Results prefix: %RESULT_DIR%\!RUN_NAME!
)
exit /b 0

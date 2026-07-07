@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CWRU few-shot noise robustness sweep for A0 SDSFNet.
REM
REM Configuration:
REM   * Dataset: CWRU 10-class, data directory = data
REM   * Few-shot training samples per class: 20, 50, 100, 200
REM   * Train SNRs: -12, -10, -8, -6, -4, -2 dB Gaussian white noise
REM   * Test SNR: matched to the training SNR for each run
REM   * Seeds: 1..5
REM   * Total tasks: 4 shot settings x 6 SNRs x 5 seeds = 120 runs
REM
REM A0 recipe:
REM   SDS front-end + Haar-LPR downsample + no-SK fusion +
REM   no-phase CSMoH H6R4+R8 all-head routing.
REM ============================================================

cd /d "%~dp0"

if not exist "logs" mkdir logs
if not exist "results" mkdir results

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "DATA_DIR=data"
set "SHOTS=20 50 100 200"
set "TRAIN_SNRS=-12 -10 -8 -6 -4 -2"
set "SEEDS=1 2 3 4 5"
set "EPOCHS=100"
set "TRAIN_NOISE=gaussian"
set "TEST_NOISE_TYPES=gaussian"

set "MASTER=%LOG_DIR%\cwru_a0_fewshot_snr_sweep_5seeds_!STAMP!_master.log"

set "DATASET_ARGS=--dataset=cwru --data_dir=%DATA_DIR% --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"
set "A0_ARGS=--ffn_type li_bottleneck --simple_down4 --simple_head --wavelet_downsample haar_lpr --pos_embedding none --no_sk_fusion --token_mixer no_phase --moh_num_heads 6 --moh_rank 4 --moh_proj_rank 8 --moh_expert_strength 0.5"

echo ============================================
echo CWRU A0 few-shot sweep started at !STAMP!
echo Dataset dir: %DATA_DIR%
echo Shots per class: %SHOTS%
echo Train/Test SNRs: %TRAIN_SNRS% dB Gaussian
echo Seeds: %SEEDS%
echo Total tasks: 120
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================
(
  echo ============================================
  echo CWRU A0 few-shot sweep started at !STAMP!
  echo Dataset dir: %DATA_DIR%
  echo Shots per class: %SHOTS%
  echo Train/Test SNRs: %TRAIN_SNRS% dB Gaussian
  echo Seeds: %SEEDS%
  echo Total tasks: 120
  echo Logs: %LOG_DIR%
  echo Results: %RESULT_DIR%
  echo ============================================
) > "!MASTER!"

for %%N in (%SHOTS%) do (
  for %%S in (%TRAIN_SNRS%) do (
    for %%E in (%SEEDS%) do (
      set "SHOT=%%N"
      set "TRAIN_SNR=%%S"
      set "SEED=%%E"
      set "RUN_NAME=cwru_a0_fewshot!SHOT!_train-gaussian_snr!TRAIN_SNR!db_test-gaussian_snr!TRAIN_SNR!db_seed!SEED!_!STAMP!"
      set "RUN_LOG=%LOG_DIR%\!RUN_NAME!.log"

      echo.
      echo [A0] shots=!SHOT! snr=!TRAIN_SNR! seed=!SEED!
      echo     Log: !RUN_LOG!
      (
        echo.
        echo [A0] shots=!SHOT! snr=!TRAIN_SNR! seed=!SEED!
        echo     Log: !RUN_LOG!
      ) >> "!MASTER!"

      python train_model.py sds_dsfb %COMMON_ARGS% %A0_ARGS% ^
        --train_samples_per_class=!SHOT! ^
        --noise_type=%TRAIN_NOISE% ^
        --train_snr_min=!TRAIN_SNR! ^
        --train_snr_max=!TRAIN_SNR! ^
        --snr_list=!TRAIN_SNR! ^
        --seed=!SEED! ^
        --run_name=!RUN_NAME! > "!RUN_LOG!" 2>&1

      if errorlevel 1 (
        echo     FAILED. See: !RUN_LOG!
        echo     FAILED. See: !RUN_LOG! >> "!MASTER!"
        exit /b 1
      )

      echo     Done. Results prefix: %RESULT_DIR%\!RUN_NAME!
      echo     Done. Results prefix: %RESULT_DIR%\!RUN_NAME! >> "!MASTER!"
    )
  )
)

echo.
echo All CWRU A0 few-shot runs finished.
echo. >> "!MASTER!"
echo All CWRU A0 few-shot runs finished. >> "!MASTER!"
exit /b 0

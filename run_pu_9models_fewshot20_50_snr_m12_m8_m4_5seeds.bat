@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM PU few-shot comparison for 9 models.
REM
REM Dataset/settings:
REM   * PU-BS-10, conditions: N15_M01_F10,N09_M07_F10,N15_M07_F04
REM   * Measurement files: 1..3
REM   * Shots per class: 20, 50
REM   * Train SNRs: -12, -8, -4 dB Gaussian
REM   * Test SNR: matched to the training SNR
REM   * Seeds: 1..5
REM
REM Models:
REM   SDSFNet, ALMformer, Convformer-NSE, DRSN-CW, GTFENet,
REM   Liconvformer, TSLANet, WDCNN, CNN-Transformer
REM
REM Note:
REM   SDSFNet 50-shot has already been run, so this script skips that
REM   combination and only runs SDSFNet 20-shot plus all missing baselines.
REM
REM Total new tasks:
REM   SDSFNet: 1 shot setting x 3 SNRs x 5 seeds = 15
REM   Other 8 models: 8 x 2 shot settings x 3 SNRs x 5 seeds = 240
REM   Total = 255 runs
REM ============================================================

cd /d "%~dp0"

if not exist "logs" mkdir logs
if not exist "results" mkdir results

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "DATA_DIR=PU_extracted"
set "SHOTS=20 50"
set "TRAIN_SNRS=-12 -8 -4"
set "SEEDS=1 2 3 4 5"
set "EPOCHS=100"
set "TRAIN_NOISE=gaussian"
set "TEST_NOISE_TYPES=gaussian"

set "MASTER=%LOG_DIR%\pu_9models_fewshot20_50_snr_m12_m8_m4_5seeds_!STAMP!_master.log"

set "DATASET_ARGS=--dataset=pu --data_dir=%DATA_DIR% --pu_condition=N15_M01_F10,N09_M07_F10,N15_M07_F04 --pu_measurement_start=1 --pu_measurement_end=3 --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"

echo ============================================================
echo PU 9-model few-shot sweep started at !STAMP!
echo Shots: %SHOTS%
echo Train/Test SNRs: %TRAIN_SNRS% dB Gaussian
echo Seeds: %SEEDS%
echo New tasks: 255
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================================
(
  echo ============================================================
  echo PU 9-model few-shot sweep started at !STAMP!
  echo Shots: %SHOTS%
  echo Train/Test SNRs: %TRAIN_SNRS% dB Gaussian
  echo Seeds: %SEEDS%
  echo New tasks: 255
  echo Logs: %LOG_DIR%
  echo Results: %RESULT_DIR%
  echo ============================================================
) > "!MASTER!"

for %%H in (%SHOTS%) do (
  for %%S in (%TRAIN_SNRS%) do (
    for %%E in (%SEEDS%) do (
      set "SHOT=%%H"
      set "TRAIN_SNR=%%S"
      set "SEED=%%E"

      call :run_model "sdsfnet" "SDSFNet" "sds_dsfb" "--ffn_type li_bottleneck --simple_down4 --simple_head --wavelet_downsample haar_lpr --pos_embedding none --no_sk_fusion --token_mixer no_phase --moh_num_heads 6 --moh_rank 4 --moh_proj_rank 8 --moh_expert_strength 0.5"
      if errorlevel 1 exit /b 1

      call :run_model "almformer" "ALMformer" "almformer" ""
      if errorlevel 1 exit /b 1

      call :run_model "convformer_nse" "Convformer-NSE" "convformer_nse" ""
      if errorlevel 1 exit /b 1

      call :run_model "drsn_cw" "DRSN-CW" "drsn_cw" ""
      if errorlevel 1 exit /b 1

      call :run_model "gtfenet" "GTFENet" "gtfenet" ""
      if errorlevel 1 exit /b 1

      call :run_model "liconvformer" "Liconvformer" "liconvformer" ""
      if errorlevel 1 exit /b 1

      call :run_model "tslanet" "TSLANet" "tslanet" ""
      if errorlevel 1 exit /b 1

      call :run_model "wdcnn" "WDCNN" "wdcnn" ""
      if errorlevel 1 exit /b 1

      call :run_model "cnn_transformer" "CNN-Transformer" "cnn_transformer" ""
      if errorlevel 1 exit /b 1
    )
  )
)

echo.
echo All PU 9-model few-shot runs finished.
echo. >> "!MASTER!"
echo All PU 9-model few-shot runs finished. >> "!MASTER!"
exit /b 0

:run_model
set "MODEL_ID=%~1"
set "MODEL_NAME=%~2"
set "MODEL_CMD=%~3"
set "MODEL_ARGS=%~4"

REM SDSFNet 50-shot results already exist from the earlier A0 run.
if /I "!MODEL_ID!"=="sdsfnet" if "!SHOT!"=="50" (
  echo.
  echo [SKIP] !MODEL_NAME! shots=!SHOT! snr=!TRAIN_SNR! seed=!SEED! already done.
  echo [SKIP] !MODEL_NAME! shots=!SHOT! snr=!TRAIN_SNR! seed=!SEED! already done. >> "!MASTER!"
  exit /b 0
)

set "RUN_NAME=pu_9models_!MODEL_ID!_fewshot!SHOT!_train-gaussian_snr!TRAIN_SNR!db_test-gaussian_snr!TRAIN_SNR!db_seed!SEED!_!STAMP!"
set "RUN_LOG=%LOG_DIR%\!RUN_NAME!.log"

echo.
echo [RUN] !MODEL_NAME! shots=!SHOT! snr=!TRAIN_SNR! seed=!SEED!
echo       Log: !RUN_LOG!
(
  echo.
  echo [RUN] !MODEL_NAME! shots=!SHOT! snr=!TRAIN_SNR! seed=!SEED!
  echo       Log: !RUN_LOG!
) >> "!MASTER!"

python train_model.py !MODEL_CMD! %COMMON_ARGS% !MODEL_ARGS! ^
  --train_samples_per_class=!SHOT! ^
  --noise_type=%TRAIN_NOISE% ^
  --train_snr_min=!TRAIN_SNR! ^
  --train_snr_max=!TRAIN_SNR! ^
  --snr_list=!TRAIN_SNR! ^
  --seed=!SEED! ^
  --run_name=!RUN_NAME! > "!RUN_LOG!" 2>&1

if errorlevel 1 (
  echo       FAILED. See: !RUN_LOG!
  echo       FAILED. See: !RUN_LOG! >> "!MASTER!"
  exit /b 1
)

echo       Done. Results prefix: %RESULT_DIR%\!RUN_NAME!
echo       Done. Results prefix: %RESULT_DIR%\!RUN_NAME! >> "!MASTER!"
exit /b 0

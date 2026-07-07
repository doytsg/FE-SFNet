@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CWRU A0 noise generalization analysis.
REM
REM Three settings are run for seeds 1..5:
REM   A) mixed noise training: Gaussian SNR sampled from -12 to 0 dB
REM      test SNRs: -12,-10,-8,-6,-4,-2,0
REM   B) clean training: no training/validation noise
REM      test SNRs: -2,0,2,4,6,8,10
REM   C) fixed -12 dB training: Gaussian SNR fixed at -12 dB
REM      test SNRs: -12,-10,-8,-6,-4,-2,0
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
set "SEEDS=1 2 3 4 5"
set "EPOCHS=100"
set "TEST_NOISE_TYPES=gaussian"
set "TEST_SNRS_NOISY=-12,-10,-8,-6,-4,-2,0"
set "TEST_SNRS_CLEAN=-2,0,2,4,6,8,10"

set "MASTER=%LOG_DIR%\cwru_a0_noise_generalization_5seeds_!STAMP!_master.log"

set "DATASET_ARGS=--dataset=cwru --data_dir=%DATA_DIR% --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --test_noise --test_noise_types=%TEST_NOISE_TYPES% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"
set "A0_ARGS=--ffn_type li_bottleneck --simple_down4 --simple_head --wavelet_downsample haar_lpr --pos_embedding none --no_sk_fusion --token_mixer no_phase --moh_num_heads 6 --moh_rank 4 --moh_proj_rank 8 --moh_expert_strength 0.5"

echo ============================================
echo CWRU A0 noise generalization started at !STAMP!
echo Dataset dir: %DATA_DIR%
echo Seeds: %SEEDS%
echo Settings: mixed_m12to0, clean_train, fixed_m12
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================
(
  echo ============================================
  echo CWRU A0 noise generalization started at !STAMP!
  echo Dataset dir: %DATA_DIR%
  echo Seeds: %SEEDS%
  echo Settings: mixed_m12to0, clean_train, fixed_m12
  echo Logs: %LOG_DIR%
  echo Results: %RESULT_DIR%
  echo ============================================
) > "!MASTER!"

for %%E in (%SEEDS%) do (
  call :run_one "mixed_m12to0" "mixed Gaussian training SNR -12..0 dB" "--train_noise --val_noise --snr_per_sample --noise_type=gaussian --train_snr_min=-12 --train_snr_max=0" "%TEST_SNRS_NOISY%" "%%E"
  if errorlevel 1 exit /b 1
  call :run_one "clean_train" "clean training" "--noise_type=gaussian" "%TEST_SNRS_CLEAN%" "%%E"
  if errorlevel 1 exit /b 1
  call :run_one "fixed_m12" "fixed Gaussian training SNR -12 dB" "--train_noise --val_noise --snr_per_sample --noise_type=gaussian --train_snr_min=-12 --train_snr_max=-12" "%TEST_SNRS_NOISY%" "%%E"
  if errorlevel 1 exit /b 1
)

echo.
echo All CWRU A0 noise generalization runs finished.
echo. >> "!MASTER!"
echo All CWRU A0 noise generalization runs finished. >> "!MASTER!"
exit /b 0

:run_one
set "SETTING=%~1"
set "DESC=%~2"
set "TRAIN_ARGS=%~3"
set "SNR_LIST=%~4"
set "SEED=%~5"
set "RUN_NAME=cwru_a0_noisegen_!SETTING!_seed!SEED!_!STAMP!"
set "RUN_LOG=%LOG_DIR%\!RUN_NAME!.log"

echo.
echo [A0] !DESC! ^| seed=!SEED!
echo     Test SNRs: !SNR_LIST!
echo     Log: !RUN_LOG!
(
  echo.
  echo [A0] !DESC! ^| seed=!SEED!
  echo     Test SNRs: !SNR_LIST!
  echo     Log: !RUN_LOG!
) >> "!MASTER!"

python train_model.py sds_dsfb %COMMON_ARGS% %A0_ARGS% !TRAIN_ARGS! --snr_list=!SNR_LIST! --seed=!SEED! --run_name=!RUN_NAME! > "!RUN_LOG!" 2>&1

if errorlevel 1 (
  echo     FAILED. See: !RUN_LOG!
  echo     FAILED. See: !RUN_LOG! >> "!MASTER!"
  exit /b 1
)

echo     Done. Results prefix: %RESULT_DIR%\!RUN_NAME!
echo     Done. Results prefix: %RESULT_DIR%\!RUN_NAME! >> "!MASTER!"
exit /b 0

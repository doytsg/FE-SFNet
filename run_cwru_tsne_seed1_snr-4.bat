@echo off
setlocal enabledelayedexpansion

REM Train 9 CWRU models at Gaussian SNR=-4 dB with seed=1, then export
REM penultimate-layer features and t-SNE coordinates for Origin plotting.
REM SDS-DSFB uses current default recipe (SK fusion on, gap cross-scale).

if not exist "logs" mkdir logs
if not exist "results" mkdir results
if not exist "tsne_outputs" mkdir tsne_outputs

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%I"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "TSNE_DIR=tsne_outputs"
set "SEED=1"
set "TRAIN_NOISE=gaussian"
set "TRAIN_SNR=-12"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=cwru --data_dir=data --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --noise_type=%TRAIN_NOISE% --train_snr_min=%TRAIN_SNR% --train_snr_max=%TRAIN_SNR% --test_snr=%TRAIN_SNR% --epochs=%EPOCHS% --results_dir=%RESULT_DIR% --seed=%SEED%"
set "EXPORT_ARGS=%DATASET_ARGS% --batch_size=128 --num_workers=0 --results_dir=%RESULT_DIR% --seed=%SEED% --noise_type=%TRAIN_NOISE% --export_snr=%TRAIN_SNR% --export_noise_type=%TRAIN_NOISE%"

echo ============================================
echo CWRU t-SNE export experiment
echo Train/Test noise: %TRAIN_NOISE% %TRAIN_SNR% dB
echo Seed: %SEED%
echo Models: 9
echo Logs: %LOG_DIR%
echo t-SNE CSV outputs: %TSNE_DIR%
echo ============================================

call :run_task "sds_dsfb" "sds_dsfb" "--dsfb_num_heads 4 --dsfb_freq_kernel_size 5 --ffn_type li_bottleneck --simple_down4 --simple_head"
if errorlevel 1 exit /b 1
call :run_task "almformer" "almformer" ""
if errorlevel 1 exit /b 1
call :run_task "convformer_nse" "convformer_nse" ""
if errorlevel 1 exit /b 1
call :run_task "drsn_cw" "drsn_cw" ""
if errorlevel 1 exit /b 1
call :run_task "gtfenet" "gtfenet" ""
if errorlevel 1 exit /b 1
call :run_task "liconvformer" "liconvformer" ""
if errorlevel 1 exit /b 1
call :run_task "tslanet" "tslanet" ""
if errorlevel 1 exit /b 1
call :run_task "wdcnn" "wdcnn" ""
if errorlevel 1 exit /b 1
call :run_task "cnn_transformer" "cnn_transformer" ""
if errorlevel 1 exit /b 1

echo.
echo ============================================
echo All training and t-SNE exports completed.
echo CSV files are in %TSNE_DIR%.
echo ============================================
exit /b 0

:run_task
set "MODEL_NAME=%~1"
set "MODEL_CMD=%~2"
set "MODEL_ARGS=%~3"
set "RUN_NAME=cwru_tsne_%MODEL_NAME%_train-gaussian_snr%TRAIN_SNR%db_seed%SEED%_%timestamp%"
set "LOG_PATH=%LOG_DIR%\%RUN_NAME%.log"
set "CKPT_PATH=%RESULT_DIR%\%RUN_NAME%_best.pth"
set "OUT_PREFIX=%TSNE_DIR%\%RUN_NAME%"

echo.
echo --------------------------------------------
echo [%MODEL_NAME%] Training...
echo --------------------------------------------
python train_model.py %MODEL_CMD% %COMMON_ARGS% --run_name=%RUN_NAME% %MODEL_ARGS% > "%LOG_PATH%" 2>&1
if errorlevel 1 (
    echo [%MODEL_NAME%] training FAILED. See: %LOG_PATH%
    exit /b 1
)

echo [%MODEL_NAME%] Exporting t-SNE CSV...
python export_tsne_features.py %MODEL_CMD% %EXPORT_ARGS% --checkpoint="%CKPT_PATH%" --output_prefix="%OUT_PREFIX%" %MODEL_ARGS% >> "%LOG_PATH%" 2>&1
if errorlevel 1 (
    echo [%MODEL_NAME%] t-SNE export FAILED. See: %LOG_PATH%
    exit /b 1
)

echo [%MODEL_NAME%] Done.
echo   Log: %LOG_PATH%
echo   Features: %OUT_PREFIX%_features.csv
echo   t-SNE: %OUT_PREFIX%_tsne.csv
exit /b 0

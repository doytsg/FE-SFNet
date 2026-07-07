@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
if not exist logs mkdir logs

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%I"
set "MASTER=logs\pu_mix_csmoh_plus_proj8_h6r4_snr_sweep_5seeds_!STAMP!_master.log"

echo CSMoH H6R4+R8 all-head sweep started at !STAMP! > "!MASTER!"

for %%S in (-8 -4 0) do (
  for /L %%E in (1,1,5) do (
    set "RUN_NAME=pu_mix_csmoh_plus_proj8_h6r4_train-gaussian_snr%%Sdb_test-gaussian_snr%%Sdb_seed%%E_!STAMP!"
    set "RUN_LOG=logs\!RUN_NAME!.log"
    echo.>> "!MASTER!"
    echo [csmoh_plus_proj8_h6r4] SNR=%%S seed=%%E>> "!MASTER!"
    echo     Log: !RUN_LOG!>> "!MASTER!"

    python train_model.py sds_dsfb ^
      --dataset=pu ^
      --data_dir=PU_extracted ^
      --pu_condition="N15_M01_F10,N09_M07_F10,N15_M07_F04" ^
      --pu_measurement_start=1 ^
      --pu_measurement_end=3 ^
      --window_size=2048 ^
      --stride=2048 ^
      --num_classes=10 ^
      --train_noise ^
      --val_noise ^
      --snr_per_sample ^
      --test_noise ^
      --test_noise_types=gaussian ^
      --snr_list=%%S ^
      --epochs=100 ^
      --results_dir=results ^
      --ffn_type li_bottleneck ^
      --simple_down4 ^
      --simple_head ^
      --wavelet_downsample haar_lpr ^
      --token_mixer csmoh_plus ^
      --moh_num_heads 6 ^
      --moh_rank 4 ^
      --moh_proj_rank 8 ^
      --moh_expert_strength 0.5 ^
      --noise_type=gaussian ^
      --train_snr_min=%%S ^
      --train_snr_max=%%S ^
      --pos_embedding=none ^
      --seed=%%E ^
      --run_name=!RUN_NAME! > "!RUN_LOG!" 2>&1

    if errorlevel 1 (
      echo     FAILED. See !RUN_LOG!>> "!MASTER!"
      exit /b 1
    )
    echo     Done.>> "!MASTER!"
  )
)

echo.>> "!MASTER!"
echo All runs finished.>> "!MASTER!"
type "!MASTER!"

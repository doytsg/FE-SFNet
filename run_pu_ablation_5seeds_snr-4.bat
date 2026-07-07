@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Ablation study for SDSFNet (no-phase CSMoH version) on
REM PU-Mix 10-class. Filename kept ("5seeds_snr-4") for backwards
REM compatibility, but the actual configuration below is:
REM     * 5 seeds (1..5)
REM     * train SNRs: -8, -4, 0 dB (Gaussian)
REM     * each run tests at the same SNR as training
REM     * 7 ablation variants enabled
REM Total tasks: 7 variants x 3 SNRs x 5 seeds = 105 runs.
REM ============================================================
REM
REM Variant inventory (all share BASE_SDSFNET, last flag wins in argparse):
REM   A0  Full SDSFNet without SK fusion -- Haar low-pass residual downsample +
REM         no-phase CSMoH h6r4+r8 with all-head routing (current lean recipe)
REM   A1  w/o CS-SDS Front-End                  (--no_sds_frontend)
REM         The A1 replacement is a *minimal* strided-conv stem
REM         (Conv1->32->64->128 + AvgPool), making A1 strictly cheaper
REM         than Full so the comparison is fair.
REM   A1B w/o CS-SDS Front-End, matched Conv   (--no_sds_frontend --matched_conv_frontend)
REM         Parameter-matched plain Conv stem with two residual separable
REM         Conv refinement blocks; no sparse dilations, no Haar, no coupling.
REM   A2  w/o wavelet-shrink downsample        (--no_haar_wavelet)
REM         A2 also implicitly disables adaptive shrinkage.
REM   A4  no-phase CSMoH -> standard MHSA      (--use_mhsa)
REM         Attention baseline that removes routed spectral heads.
REM   A5  learned positional embedding         (--pos_embedding learned)
REM         Default recipe uses no additive PE; this checks whether a learned
REM         PE still helps.
REM   A6  minimal baseline:                    (--no_sds_frontend --use_mhsa)
REM         w/o CS-SDS Front-End AND w/o no-phase CSMoH (both major
REM         contributions removed simultaneously).
REM ------------------------------------------------------------
REM Removed from previous exploratory revisions: SK fusion, complex phase,
REM two-stage, band-style, and internal no_xxx module switches.
REM ============================================================

if not exist "logs" mkdir logs
if not exist "results" mkdir results

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "timestamp=%%I"

set "RESULT_DIR=results"
set "LOG_DIR=logs"
set "SEEDS=1 2 3 4 5"
set "TRAIN_NOISE=gaussian"
set "TRAIN_SNRS=-12"
set "TEST_NOISE_TYPES=gaussian"
set "EPOCHS=100"

set "DATASET_ARGS=--dataset=pu --data_dir=PU_extracted --pu_condition=N15_M01_F10,N09_M07_F10,N15_M07_F04 --pu_measurement_start=1 --pu_measurement_end=3 --window_size=2048 --stride=2048 --num_classes=10"
set "COMMON_ARGS=%DATASET_ARGS% --train_noise --val_noise --snr_per_sample --test_noise --test_noise_types=%TEST_NOISE_TYPES% --epochs=%EPOCHS% --results_dir=%RESULT_DIR%"

REM Default SDSFNet recipe shared by every variant.
REM SK fusion is disabled globally; no separate w/o-SK ablation is needed.
REM Per-variant overrides are appended after this base, so they win in argparse.
set "BASE_SDSFNET=--ffn_type li_bottleneck --simple_down4 --simple_head --wavelet_downsample haar_lpr --pos_embedding none --no_sk_fusion --token_mixer no_phase --moh_num_heads 6 --moh_rank 4 --moh_proj_rank 8 --moh_expert_strength 0.5"

echo ============================================
echo SDSFNet ablation (no-phase CSMoH h6r4+r8 all-head routing, no SK fusion)
echo Dataset: PU-Mix 10-class
echo Seeds: %SEEDS%
echo Train SNRs: %TRAIN_SNRS% dB (Gaussian)
echo Test SNR: matched to each training SNR
echo Variants: A0 A1 A1B A2 A4 A5 A6  (7 total, all use --no_sk_fusion)
echo Total tasks: 7 variants x 3 SNRs x 5 seeds = 105 runs
echo Logs: %LOG_DIR%
echo Results: %RESULT_DIR%
echo ============================================

for %%S in (%TRAIN_SNRS%) do (
    set "TRAIN_SNR=%%S"

    for %%D in (%SEEDS%) do (
        set "SEED=%%D"

        echo.
        echo --------------------------------------------
        echo SNR=%%S dB, Seed=%%D: 7 ablation variants
        echo --------------------------------------------

        call :run_variant "a0_full"          ""
        call :run_variant "a1_no_sds"        "--no_sds_frontend"
        call :run_variant "a1b_matched_conv" "--no_sds_frontend --matched_conv_frontend"
        call :run_variant "a2_no_haar"       "--no_haar_wavelet"
        call :run_variant "a4_mhsa"          "--use_mhsa"
        call :run_variant "a5_learned_pos"   "--pos_embedding learned"
        call :run_variant "a6_minimal"       "--no_sds_frontend --use_mhsa"

        echo SNR=%%S dB, Seed=%%D completed.
    )
)

echo.
echo ============================================
echo No-phase CSMoH ablation completed.
echo Logs saved in %LOG_DIR%.
echo Result files saved in %RESULT_DIR%.
echo ============================================
exit /b 0

:run_variant
set "VARIANT=%~1"
set "OVERRIDES=%~2"
set "RUN_NAME=pu_mix_sds_dsfb_ablation_%VARIANT%_train-gaussian_snr!TRAIN_SNR!db_seed!SEED!_%timestamp%"
set "LOG_PATH=%LOG_DIR%\!RUN_NAME!.log"

REM Per-variant overrides go AFTER %BASE_SDSFNET% so they win in argparse.
REM Test SNR is matched to the current training SNR via --snr_list=!TRAIN_SNR!.
echo   [%VARIANT%] snr=!TRAIN_SNR! seed=!SEED!
python train_model.py sds_dsfb %COMMON_ARGS% %BASE_SDSFNET% --noise_type=%TRAIN_NOISE% --train_snr_min=!TRAIN_SNR! --train_snr_max=!TRAIN_SNR! --snr_list=!TRAIN_SNR! --seed=!SEED! --run_name=!RUN_NAME! %OVERRIDES% > "!LOG_PATH!" 2>&1
if errorlevel 1 (
    echo       FAILED. See: !LOG_PATH!
) else (
    echo       Done. Log: !LOG_PATH!
    echo       Results prefix: %RESULT_DIR%\!RUN_NAME!
)
exit /b 0

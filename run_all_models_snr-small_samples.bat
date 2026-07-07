@echo off
chcp 65001 >nul
if not exist logs mkdir logs

echo ============================================
echo Running All Models - Multiple SNR Levels (10 samples per class)
echo ============================================

REM SNR = -12 dB
echo.
echo [SNR=-12dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr-12_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr-12_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr-12_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr-12_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr-12_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr-12_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr-12_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr-12_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr-12_10samples.log 2>&1
echo Done!


REM SNR = -10 dB
echo.
echo [SNR=-10dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr-10_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr-10_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr-10_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr-10_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr-10_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr-10_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr-10_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr-10_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr-10_10samples.log 2>&1
echo Done!

REM SNR = -8 dB
echo.
echo [SNR=-8dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr-8_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr-8_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr-8_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr-8_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr-8_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr-8_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr-8_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr-8_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr-8_10samples.log 2>&1
echo Done!

REM SNR = -6 dB
echo.
echo [SNR=-6dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr-6_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr-6_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr-6_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr-6_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr-6_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr-6_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr-6_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr-6_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr-6_10samples.log 2>&1
echo Done!

REM SNR = -4 dB
echo.
echo [SNR=-4dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr-4_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr-4_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr-4_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr-4_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr-4_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr-4_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr-4_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr-4_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr-4_10samples.log 2>&1
echo Done!

REM SNR = -2 dB
echo.
echo [SNR=-2dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr-2_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr-2_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr-2_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr-2_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr-2_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr-2_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr-2_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr-2_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr-2_10samples.log 2>&1
echo Done!

REM SNR = 0 dB
echo.
echo [SNR=0dB] Running 9 models...
python GTFENET.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\gtfenet_snr0_10samples.log 2>&1
python train_mgstl_transformer_1_12.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --use_asb --use_icb --epochs 150 --train_samples_per_class 100 > logs\mslk_snr0_10samples.log 2>&1
python DRSN-CW.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\drsn_cw_snr0_10samples.log 2>&1
python train_liconvformer.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\liconvformer_snr0_10samples.log 2>&1
python ALMformer.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\almformer_snr0_10samples.log 2>&1
python train_convformer_nse.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\convformer_nse_snr0_10samples.log 2>&1
python TSLANet_classification.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\tslanet_snr0_10samples.log 2>&1
python WDCNN.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\wdcnn_snr0_10samples.log 2>&1
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min 0 --train_snr_max 0 --snr_per_sample --consistency_weight 0 --test_noise --epochs 150 --train_samples_per_class 100 > logs\cnn_transformer_snr0_10samples.log 2>&1
echo Done!

echo.
echo ============================================
echo All experiments completed! Total: 54 tasks
echo Logs saved in logs directory
echo ============================================
pause

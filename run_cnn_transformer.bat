@echo off
chcp 65001 >nul

if not exist logs mkdir logs

echo ============================================
echo CNN Transformer Experiments
echo ============================================

echo [1/6] SNR=-12dB...
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -12 --train_snr_max -12 --snr_per_sample --consistency_weight 0 --test_noise --epochs 100 > logs\cnn_transformer_snr-12.log 2>&1
echo Done!

echo [2/6] SNR=-10dB...
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -10 --train_snr_max -10 --snr_per_sample --consistency_weight 0 --test_noise --epochs 100 > logs\cnn_transformer_snr-10.log 2>&1
echo Done!

echo [3/6] SNR=-8dB...
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -8 --train_snr_max -8 --snr_per_sample --consistency_weight 0 --test_noise --epochs 100 > logs\cnn_transformer_snr-8.log 2>&1
echo Done!

echo [4/6] SNR=-6dB...
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -6 --train_snr_max -6 --snr_per_sample --consistency_weight 0 --test_noise --epochs 100 > logs\cnn_transformer_snr-6.log 2>&1
echo Done!

echo [5/6] SNR=-4dB...
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -4 --train_snr_max -4 --snr_per_sample --consistency_weight 0 --test_noise --epochs 100 > logs\cnn_transformer_snr-4.log 2>&1
echo Done!

echo [6/6] SNR=-2dB...
python train_cnn_transformer.py --train_noise --val_noise --train_snr_min -2 --train_snr_max -2 --snr_per_sample --consistency_weight 0 --test_noise --epochs 100 > logs\cnn_transformer_snr-2.log 2>&1
echo Done!

echo ============================================
echo All experiments completed!
echo ============================================
pause

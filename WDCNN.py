"""
WDCNN: Wide Deep Convolutional Neural Network for Bearing Fault Diagnosis
"""

import argparse

import torch

from models.wdcnn import WDCNN
from train_common import (
    ExperimentConfig,
    add_common_args,
    maybe_report_model_stats,
    run_experiment,
    set_seed,
)

try:
    from thop import clever_format, profile
    THOP_AVAILABLE = True
except ImportError:
    clever_format = None
    profile = None
    THOP_AVAILABLE = False
    print("Warning: thop not installed. Install with 'pip install thop' for FLOPs calculation.")


def build_parser():
    parser = argparse.ArgumentParser(description="WDCNN for CWRU Bearing Fault Diagnosis (10-class)")
    add_common_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in classifier")
    parser.add_argument("--no_dropout", action="store_true", help="Disable dropout")
    parser.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10,12")
    return parser


def build_model(args):
    use_dropout = not args.no_dropout
    model = WDCNN(num_classes=10, use_dropout=use_dropout, dropout=args.dropout)
    return model, use_dropout


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("WDCNN: Wide Deep Convolutional Neural Network")
    print("=" * 60)

    model, use_dropout = build_model(args)
    model = model.to(device)
    print(f"\nModel: WDCNN (use_dropout={use_dropout}, dropout={args.dropout})")
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="wdcnn",
        model_display_name="WDCNN",
        confusion_title="Confusion Matrix - WDCNN",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="WDCNN",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()
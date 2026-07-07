"""
GTFENET: Gramian-Time-Frequency Enhanced Network
"""

import argparse

import torch

from models.gtfenet import GTFENET
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
    parser = argparse.ArgumentParser(description="GTFENET for CWRU bearing fault diagnosis (10-class)")
    add_common_args(parser)
    parser.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10")
    return parser


def build_model(args):
    return GTFENET(num_classes=10, input_length=args.window_size)


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="gtfenet",
        model_display_name="GTFENET",
        confusion_title="Confusion Matrix - GTFENET",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="GTFENET",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

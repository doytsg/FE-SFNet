import argparse

import torch

from models.convformer_nse import ConvformerNSE
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


def main():
    parser = argparse.ArgumentParser(description="Convformer-NSE for CWRU (10-class)")
    add_common_args(parser)
    parser.set_defaults(snr_list="-12,-11,-10,-8,-6,-5,-4,-2,0,2,4,6,8,10,12")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("Convformer-NSE Training Script")
    print("=" * 60)

    model = ConvformerNSE(in_channels=1, num_classes=10).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="convformer_nse",
        model_display_name="Convformer-NSE",
        confusion_title="Confusion Matrix - Convformer-NSE",
        noise_plot_title="Convformer-NSE Robustness Under Different Noise Levels",
        history_title_suffix="Convformer-NSE",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

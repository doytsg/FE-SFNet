import argparse

import torch

from models.liconvformer import Liconvformer
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
    parser = argparse.ArgumentParser(description="Liconvformer for CWRU (train/eval like train_e2stformer.py)")
    add_common_args(parser)
    parser.add_argument("--dim", type=int, default=16,
                        help="Base dim; final embedding is 8*dim. Use 16 to get d_model=128.")
    parser.add_argument("--drop", type=float, default=0.1)
    return parser


def build_model(args):
    return Liconvformer(
        None,
        in_channel=1,
        out_channel=10,
        drop=args.drop,
        dim=args.dim,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="liconvformer",
        model_display_name="Liconvformer",
        confusion_title="Confusion Matrix - Liconvformer",
        noise_plot_title="Model Robustness Under Different Noise Levels",
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

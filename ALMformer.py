import argparse

import torch

from models.almformer import ALMformer
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
    parser = argparse.ArgumentParser(description="ALMformer for CWRU (10-class)")
    add_common_args(parser)
    parser.add_argument("--num_classes", type=int, default=10, help="分类数")
    parser.add_argument("--depth", type=int, default=4, help="MetaFormer Blocks 数量")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10")
    return parser


def build_model(args):
    return ALMformer(num_classes=args.num_classes, depth=args.depth)


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("ALMformer Training")
    print("=" * 60)

    model = build_model(args).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="almformer",
        model_display_name="ALMformer",
        confusion_title="Confusion Matrix - ALMformer",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="ALMformer",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()
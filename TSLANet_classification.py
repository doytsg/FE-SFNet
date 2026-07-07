"""
TSLANet Classification - Standalone Training Script
"""

import argparse

import torch

from models.tslanet import TSLANet
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
    parser = argparse.ArgumentParser(description="TSLANet for CWRU (10-class)")
    add_common_args(parser)
    parser.add_argument("--num_classes", type=int, default=10, help="分类数")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=2, help="Number of TSLANet layers")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_icb", action="store_true", default=True,
                        help="Use ICB (Interactive Convolutional Block)")
    parser.add_argument("--no_icb", action="store_true", help="Disable ICB")
    parser.add_argument("--use_asb", action="store_true", default=True,
                        help="Use ASB (Adaptive Spectral Block)")
    parser.add_argument("--no_asb", action="store_true", help="Disable ASB")
    parser.add_argument("--adaptive_filter", action="store_true", default=True,
                        help="Use adaptive filter in ASB")
    parser.add_argument("--no_adaptive_filter", action="store_true",
                        help="Disable adaptive filter in ASB")
    return parser


def resolve_flags(args):
    use_icb = args.use_icb and not args.no_icb
    use_asb = args.use_asb and not args.no_asb
    adaptive_filter = args.adaptive_filter and not args.no_adaptive_filter
    return use_icb, use_asb, adaptive_filter


def build_model(args):
    use_icb, use_asb, adaptive_filter = resolve_flags(args)
    model = TSLANet(
        seq_len=args.window_size,
        num_channels=1,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        dropout_rate=args.dropout,
        use_icb=use_icb,
        use_asb=use_asb,
        adaptive_filter=adaptive_filter,
    )
    return model, use_icb, use_asb, adaptive_filter


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, use_icb, use_asb, adaptive_filter = build_model(args)
    model = model.to(device)

    print("=" * 60)
    print("TSLANet Training")
    print("=" * 60)
    print(f"[INFO] ICB: {use_icb}, ASB: {use_asb}, Adaptive Filter: {adaptive_filter}")

    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="tslanet",
        model_display_name="TSLANet",
        confusion_title="Confusion Matrix - TSLANet",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="TSLANet",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

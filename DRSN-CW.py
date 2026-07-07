#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch

from models.drsn_cw import DRSN_CW, DRSN_CW_Lite
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
    parser = argparse.ArgumentParser(description="DRSN-CW for CWRU (10-class)")
    add_common_args(parser)
    parser.add_argument("--num_classes", type=int, default=10, help="分类数")
    parser.add_argument("--base_channels", type=int, default=32, help="基础通道数")
    parser.add_argument("--lite", action="store_true", help="使用轻量版模型")
    parser.set_defaults(snr_list="-12,-10,-9,-8,-6,-5,-4,-2,0,2,4,6,8,10")
    return parser


def build_model(args):
    if args.lite:
        return DRSN_CW_Lite(
            num_classes=args.num_classes,
            in_channels=1,
            base_channels=args.base_channels,
            num_blocks=[1, 1, 1, 1],
        )
    return DRSN_CW(
        num_classes=args.num_classes,
        in_channels=1,
        base_channels=args.base_channels,
        num_blocks=[2, 2, 2, 2],
    )


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 60)
    print("DRSN-CW Training (Deep Residual Shrinkage Network)")
    print("=" * 60)
    print(f"[INFO] Model: {'DRSN-CW-Lite' if args.lite else 'DRSN-CW'}")
    print(f"[INFO] Base channels: {args.base_channels}")

    model = build_model(args).to(device)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="drsn_cw",
        model_display_name="DRSN-CW",
        confusion_title="Confusion Matrix - DRSN-CW",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="DRSN-CW",
        classification_zero_division=0,
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

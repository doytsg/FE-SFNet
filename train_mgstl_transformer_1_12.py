import argparse

import torch

from models.mslk_transformer import MSLKTransformer
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
    parser = argparse.ArgumentParser(
        description="MSLK Transformer (Ablation: Multi-Scale Large Kernel Only) for CWRU (10-class)",
    )
    add_common_args(parser)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel_sizes", type=str, default="15,31,51",
                        help="Comma-separated kernel sizes for MG-STL blocks")
    parser.add_argument("--no_gated_sk", action="store_true",
                        help="消融实验：禁用SK融合，使用原始Concat融合")
    parser.add_argument("--use_asb", action="store_true",
                        help="使用 DSFB (动态频谱融合块) 替代多头自注意力机制")
    parser.add_argument("--use_icb", action="store_true",
                        help="使用 ICB (交互式卷积块) 替代标准 FFN (仅在 use_asb 时有效)")
    parser.add_argument("--channel_mixer_groups", type=int, default=4,
                        help="Adaptive_Spectral_Block 中频域 channel_mixer 的分组数（建议 4 或 8）")
    parser.add_argument("--no_channel_mixer", action="store_true",
                        help="消融实验: 禁用 DSFB 中的频域 Channel Mixer")
    parser.set_defaults(snr_list="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10,12")
    return parser


def parse_kernel_sizes(kernel_sizes_text: str):
    return [int(k) for k in kernel_sizes_text.split(",")]


def build_model(args):
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)
    model = MSLKTransformer(
        num_classes=10,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.n_blocks,
        dropout=args.dropout,
        kernel_sizes=kernel_sizes,
        use_gated_sk=not args.no_gated_sk,
        use_asb=args.use_asb,
        use_icb=args.use_icb,
        channel_mixer_groups=args.channel_mixer_groups,
        use_channel_mixer=not args.no_channel_mixer,
    )
    return model, kernel_sizes


def print_model_mode(args, kernel_sizes):
    print(f"[ABLATION] Multi-Scale Large Kernel sizes: {kernel_sizes}")
    print("[ABLATION] GTG and Soft Thresholding are REMOVED in this version")

    if args.no_gated_sk:
        print("[ABLATION] 使用原始 MSLK Block (Concat融合)")
    else:
        print("[INFO] 使用 MS-SK Block (DepthwiseConv + SelectiveKernelFusion)")

    if args.use_asb:
        print("[INFO] 使用 DSFB (动态频谱融合块) 替代多头自注意力")
        print("[INFO] DSFB 特性: 跨通道交互 + 软谱注意力 + 频率特异性滤波")
        if args.use_icb:
            print("[INFO] 使用 ICB (交互式卷积块) 替代标准 FFN")
        else:
            print("[INFO] 使用标准 FFN")
    else:
        print("[INFO] 使用标准 Transformer (多头自注意力)")
        if args.use_icb:
            print("[WARNING] --use_icb 仅在 --use_asb 模式下有效，当前被忽略")


def main():
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, kernel_sizes = build_model(args)
    model = model.to(device)

    print_model_mode(args, kernel_sizes)
    maybe_report_model_stats(model, args.window_size, device, THOP_AVAILABLE, profile, clever_format)

    config = ExperimentConfig(
        model_key="mslk_ablation",
        model_display_name="MSLK Ablation",
        confusion_title="Confusion Matrix - MSLK Ablation (No GTG/Soft-Thresh)",
        noise_plot_title="Model Robustness Under Different Noise Levels",
        history_title_suffix="MSLK Ablation",
        classification_zero_division=0,
        best_model_filename="best_1_12.pth",
        history_plot_filename="training_history_mslk_ablation.png",
        noise_plot_filename="noise_robustness_mslk_ablation.png",
        confusion_matrix_filename="confusion_matrix_mslk_ablation.png",
        history_csv_template="results/training_history_mslk_snr{train_snr_min}_{train_snr_max}.csv",
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()
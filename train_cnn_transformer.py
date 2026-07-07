import argparse
import torch
from models.cnn_transformer import CNNTransformer
from train_common import ExperimentConfig, add_common_args, run_experiment, set_seed


def build_parser():
    parser = argparse.ArgumentParser(
        description="CNN-Transformer baseline for CWRU (10-class), train/eval like train_e2stformer.py",
    )
    add_common_args(parser)
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def build_model(args):
    return CNNTransformer(
        num_classes=10,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.n_blocks,
        dropout=args.dropout,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(args).to(device)
    config = ExperimentConfig(
        model_key="cnn_transformer",
        model_display_name="CNN-Transformer",
        confusion_title="Confusion Matrix",
        noise_plot_title="Model Robustness Under Different Noise Levels",
    )
    run_experiment(args, model, device, config)


if __name__ == "__main__":
    main()

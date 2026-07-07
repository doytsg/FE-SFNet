import csv
import math
import os
import random
from dataclasses import dataclass, replace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset, random_split

from cwru_dataset import (
    CWRUDataset,
    PUBS10Dataset,
    PUA2RDataset,
    PU_BS_10_CLASS_NAMES,
    PU_3CLASS_NAMES,
    PU_DEFAULT_CONDITIONS_TEXT,
)


CWRU_CLASS_NAMES = [
    "Normal",
    "Ball-007", "Ball-014", "Ball-021",
    "IR-007", "IR-014", "IR-021",
    "OR-007", "OR-014", "OR-021",
]

CLASS_NAMES = PU_BS_10_CLASS_NAMES


@dataclass(frozen=True)
class ExperimentConfig:
    model_key: str
    model_display_name: str
    confusion_title: str
    noise_plot_title: str
    history_title_suffix: str = ""
    classification_zero_division: Optional[int] = None
    best_model_filename: Optional[str] = None
    history_plot_filename: Optional[str] = None
    noise_plot_filename: Optional[str] = None
    confusion_matrix_filename: Optional[str] = None
    history_csv_template: Optional[str] = None

    @property
    def best_model_path(self) -> str:
        return self.best_model_filename or f"best_{self.model_key}.pth"

    @property
    def history_plot_path(self) -> str:
        return self.history_plot_filename or f"training_history_{self.model_key}.png"

    @property
    def noise_plot_path(self) -> str:
        return self.noise_plot_filename or f"noise_robustness_{self.model_key}.png"

    @property
    def confusion_matrix_path(self) -> str:
        return self.confusion_matrix_filename or f"confusion_matrix_{self.model_key}.png"

    def history_csv_path(self, train_snr_min: float, train_snr_max: float) -> str:
        if self.history_csv_template:
            return self.history_csv_template.format(
                train_snr_min=train_snr_min,
                train_snr_max=train_snr_max,
                model_key=self.model_key,
            )
        return f"results/training_history_{self.model_key}_snr{train_snr_min}_{train_snr_max}.csv"


def _safe_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe.strip("._-") or "run"


def _with_run_output_paths(args, config: ExperimentConfig) -> ExperimentConfig:
    results_dir = getattr(args, "results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    run_name = getattr(args, "run_name", None)
    if not run_name:
        return config

    base = os.path.join(results_dir, _safe_run_name(run_name))
    return replace(
        config,
        best_model_filename=f"{base}_best.pth",
        history_plot_filename=f"{base}_history.png",
        noise_plot_filename=f"{base}_noise_robustness.png",
        confusion_matrix_filename=f"{base}_confusion_matrix.png",
        history_csv_template=f"{base}_history.csv",
    )


def add_common_args(parser):
    parser.add_argument("--dataset", type=str, default="pu", choices=("pu", "pu_a2r", "cwru"),
                        help="数据集类型：pu 使用 PU-BS-10；pu_a2r 使用人工/真实 3 分类；cwru 使用原 CWRU 10 类")
    parser.add_argument("--pu_train_domain", type=str, default="artificial",
                        choices=("artificial", "real"),
                        help="PU-A2R 训练集来源：artificial=人工损伤；real=真实损伤")
    parser.add_argument("--pu_test_domain", type=str, default="real",
                        choices=("artificial", "real"),
                        help="PU-A2R 测试集来源：artificial=人工损伤；real=真实损伤")
    parser.add_argument("--pu_max_bearings_per_class", type=int, default=None,
                        help="PU-A2R: 每个故障类最多使用多少个轴承编号 (None=全部)；用来缩小数据量")
    parser.add_argument("--pu_train_max_bearings_per_class", type=int, default=None,
                        help="PU-A2R: training domain per-class bearing limit; overrides --pu_max_bearings_per_class")
    parser.add_argument("--pu_test_max_bearings_per_class", type=int, default=None,
                        help="PU-A2R: test domain per-class bearing limit; overrides --pu_max_bearings_per_class")
    parser.add_argument("--data_dir", type=str, default="PU_extracted", help="数据集目录，包含所有数据")
    parser.add_argument("--pu_condition", "--pu_conditions", dest="pu_condition",
                        type=str, default=PU_DEFAULT_CONDITIONS_TEXT,
                        help="PU 工况代码，支持逗号分隔多个工况；默认使用 ALMFormer 风格的三个代表工况")
    parser.add_argument("--pu_test_condition", "--pu_test_conditions", dest="pu_test_condition",
                        type=str, default=None,
                        help="PU 跨工况评估：测试集工况（leave-one-condition-out）；不设置则与训练工况一致")
    parser.add_argument("--pu_test_measurement_start", type=int, default=None,
                        help="PU 跨工况：测试集测量序号起点；默认与 pu_measurement_start 相同")
    parser.add_argument("--pu_test_measurement_end", type=int, default=None,
                        help="PU 跨工况：测试集测量序号终点；默认与 pu_measurement_end 相同")
    parser.add_argument("--pu_channel", type=str, default="vibration_1",
                        help="PU .mat 中用于分类的信号通道")
    parser.add_argument("--pu_measurement_start", type=int, default=1,
                        help="PU 测量序号起点，默认 1")
    parser.add_argument("--pu_measurement_end", type=int, default=3,
                        help="PU 测量序号终点，默认 3")
    parser.add_argument("--no_file_split", action="store_true",
                        help="PU 数据默认按测量文件分组划分；设置后改回窗口级随机划分")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    parser.add_argument("--train_samples_per_class", type=int, default=None,
                        help="小样本实验时每类的训练样本数，设置后覆盖 train_ratio")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="分类数；PU-BS-10 / CWRU 默认 10，PU-A2R 应设 3")
    parser.add_argument("--window_size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for the shared CrossEntropyLoss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for result files such as CSV, plots, and checkpoints")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Unique run name used as the prefix for all result files")

    parser.add_argument("--train_noise", action="store_true",
                        help="Add noise during training for data augmentation")
    parser.add_argument("--val_noise", action="store_true",
                        help="Add noise during validation (uses same SNR range as training)")
    parser.add_argument("--train_snr_min", type=float, default=-2,
                        help="Minimum SNR (dB) for training/validation noise")
    parser.add_argument("--train_snr_max", type=float, default=10,
                        help="Maximum SNR (dB) for training/validation noise")
    parser.add_argument("--snr_per_sample", action="store_true",
                        help="Sample SNR per sample instead of per batch")
    parser.add_argument("--noise_type", type=str, default="gaussian",
                        choices=NOISE_TYPES,
                        help="Noise type used for training/validation augmentation. "
                             "Options: clean, gaussian (default), laplace, uniform, impulse, mixed")
    parser.add_argument("--test_noise", action="store_true",
                        help="Evaluate model robustness under different noise levels")
    parser.add_argument("--test_snr", type=float, default=None,
                        help="Fixed SNR (dB) for test evaluation. If not set, uses clean data.")
    parser.add_argument("--test_noise_types", type=str, default="gaussian",
                        help="Comma-separated noise types to sweep during robustness evaluation. "
                             "Each type is curve in the noise_robustness plot.")
    parser.add_argument("--snr_list", type=str, default="-12,-11,-10,-9,-8,-7,-6,-5,-4,-2,0,2,4,6,8,10",
                        help="Comma-separated SNR values for robustness evaluation")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


NOISE_TYPES = ("clean", "gaussian", "laplace", "uniform", "impulse", "mixed")


def _unit_variance_noise(noise_type: str, signal: torch.Tensor) -> torch.Tensor:
    """Return a noise tensor with the same shape as ``signal`` and unit variance.

    Each base distribution is rescaled so that the time-axis variance is
    approximately 1, allowing :func:`add_noise_snr` to control the noise
    magnitude purely via the requested SNR.
    """
    if noise_type == "gaussian":
        return torch.randn_like(signal)
    if noise_type == "laplace":
        # Laplace inverse-CDF sampling. Variance of Laplace(0, b) is 2 * b**2,
        # so b = 1/sqrt(2) gives unit variance.
        u = torch.rand_like(signal) - 0.5
        b = 1.0 / math.sqrt(2.0)
        return -b * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp(max=1.0 - 1e-7))
    if noise_type == "uniform":
        # Uniform[-sqrt(3), sqrt(3)] has unit variance.
        return (torch.rand_like(signal) * 2.0 - 1.0) * math.sqrt(3.0)
    if noise_type == "impulse":
        return _impulse_noise(signal)
    if noise_type == "mixed":
        return _mixed_noise(signal)
    raise ValueError(f"Unsupported noise_type: {noise_type}")

def _impulse_noise(signal: torch.Tensor, density: float = 0.05) -> torch.Tensor:
    """Sparse +/-spike noise: ``density`` fraction of samples are non-zero.

    Normalised to unit variance so SNR control still works.
    """
    mask = (torch.rand_like(signal) < density).to(signal.dtype)
    sign = torch.where(
        torch.rand_like(signal) < 0.5,
        torch.ones_like(signal),
        -torch.ones_like(signal),
    )
    raw = mask * sign
    std = raw.std(dim=-1, keepdim=True).clamp(min=1e-6)
    return raw / std


def _mixed_noise(signal: torch.Tensor) -> torch.Tensor:
    """Per-sample random pick from the basic noise families."""
    options = ("gaussian", "laplace", "uniform", "impulse")
    if signal.dim() <= 1:
        return _unit_variance_noise(random.choice(options), signal)
    out = torch.empty_like(signal)
    for i in range(signal.shape[0]):
        out[i] = _unit_variance_noise(random.choice(options), signal[i])
    return out


def add_noise_snr(signal: torch.Tensor, snr_db, noise_type: str = "gaussian") -> torch.Tensor:
    """Inject noise of ``noise_type`` at the requested SNR.

    ``noise_type='clean'`` returns the signal unchanged.
    """
    if noise_type == "clean":
        return signal
    signal_power = torch.mean(signal ** 2, dim=-1, keepdim=True)
    snr_db_tensor = torch.as_tensor(snr_db, device=signal.device, dtype=signal.dtype)
    snr_linear = torch.pow(10.0, snr_db_tensor / 10.0)
    noise_power = signal_power / snr_linear
    base = _unit_variance_noise(noise_type, signal)
    return signal + base * torch.sqrt(noise_power)


def add_noise_random_snr(signal: torch.Tensor, snr_range: tuple = (-2, 10),
                         per_sample: bool = False,
                         noise_type: str = "gaussian") -> torch.Tensor:
    """Sample SNR uniformly from ``snr_range`` and call :func:`add_noise_snr`."""
    if per_sample:
        u = torch.rand(signal.size(0), 1, 1, device=signal.device, dtype=signal.dtype)
        snr_db = snr_range[0] + (snr_range[1] - snr_range[0]) * u
    else:
        snr_db = snr_range[0] + (snr_range[1] - snr_range[0]) * random.random()
    return add_noise_snr(signal, snr_db, noise_type=noise_type)


def add_gaussian_noise_snr(signal: torch.Tensor, snr_db) -> torch.Tensor:
    """Backward-compatible wrapper kept for older standalone scripts."""
    return add_noise_snr(signal, snr_db, noise_type="gaussian")


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-6)
    return x


def train_one_epoch(model, loader, optimizer, criterion, device,
                    add_noise: bool = False, snr_range: tuple = (-2, 10),
                    snr_per_sample: bool = False, noise_type: str = "gaussian"):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = normalize_batch(x.to(device))
        y = y.to(device)

        optimizer.zero_grad()
        if add_noise:
            x = add_noise_random_snr(x, snr_range, per_sample=snr_per_sample,
                                     noise_type=noise_type)
        logits = model(x)
        loss = criterion(logits, y)
        aux_loss_fn = getattr(model, "aux_loss", None)
        if callable(aux_loss_fn):
            aux_loss = aux_loss_fn()
            if torch.is_tensor(aux_loss):
                loss = loss + aux_loss.to(device=loss.device, dtype=loss.dtype)
            elif aux_loss:
                loss = loss + loss.new_tensor(float(aux_loss))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    return running_loss / len(loader.dataset), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, snr_db: float = None,
             add_noise: bool = False, snr_range: tuple = None,
             snr_per_sample: bool = False, noise_type: str = "gaussian"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = normalize_batch(x.to(device))
        y = y.to(device)

        if snr_db is not None:
            x = add_noise_snr(x, snr_db, noise_type=noise_type)
        elif add_noise and snr_range is not None:
            x = add_noise_random_snr(x, snr_range, per_sample=snr_per_sample,
                                     noise_type=noise_type)

        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = torch.argmax(logits, dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    return running_loss / len(loader.dataset), 100.0 * correct / total


@torch.no_grad()
def evaluate_full(model, loader, device, snr_db: float = None,
                  noise_type: str = "gaussian"):
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = normalize_batch(x.to(device))
        if snr_db is not None:
            x = add_noise_snr(x, snr_db, noise_type=noise_type)

        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(pred.tolist())
        all_labels.extend(y.numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def _iter_moh_routing_modules(model):
    for name, module in model.named_modules():
        if (
            hasattr(module, "set_routing_stats_enabled")
            and hasattr(module, "reset_routing_stats")
            and hasattr(module, "get_routing_stats")
        ):
            yield name, module


def _format_head_values(values, scale: float = 100.0) -> str:
    return ", ".join(f"h{i}={v * scale:.1f}%" for i, v in enumerate(values))


def _routing_cv(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 1e-12:
        return 0.0
    return float(arr.std() / mean)


@torch.no_grad()
def report_moh_routing_stats(model, loader, device, label: str,
                             snr_db: float = None, noise_type: str = "gaussian"):
    modules = list(_iter_moh_routing_modules(model))
    if not modules:
        return

    for _, module in modules:
        module.reset_routing_stats()
        module.set_routing_stats_enabled(True)

    model.eval()
    for x, _ in loader:
        x = normalize_batch(x.to(device))
        if snr_db is not None:
            x = add_noise_snr(x, snr_db, noise_type=noise_type)
        model(x)

    for _, module in modules:
        module.set_routing_stats_enabled(False)

    print(f"\n=== MoH Routing Statistics ({label}) ===")
    for name, module in modules:
        stats = module.get_routing_stats()
        if not stats:
            continue
        load = stats["load"]
        importance = stats["importance"]
        weight = stats["weight"]
        print(
            f"  {name}: heads={stats['num_heads']}, "
            f"tokens={stats['tokens']}, load_cv={_routing_cv(load):.3f}"
        )
        print(f"    selected_load: {_format_head_values(load)}")
        print(f"    routing_weight: {_format_head_values(weight)}")
        print(f"    router_importance: {_format_head_values(importance)}")


def stratified_few_shot_split(labels: np.ndarray, train_samples_per_class: int,
                              val_ratio: float, test_ratio: float, seed: int = 42):
    if train_samples_per_class <= 0:
        raise ValueError("train_samples_per_class must be positive.")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative.")

    remaining_ratio = val_ratio + test_ratio
    if remaining_ratio <= 0:
        raise ValueError("val_ratio and test_ratio cannot both be zero when using few-shot mode.")

    val_share = val_ratio / remaining_ratio
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    train_idx, val_idx, test_idx = [], [], []
    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        n_train = min(train_samples_per_class, len(cls_indices))
        n_remaining = len(cls_indices) - n_train
        n_val = int(round(n_remaining * val_share))
        n_val = min(n_val, n_remaining)
        n_test = n_remaining - n_val

        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def stratified_group_split(labels: np.ndarray, groups: np.ndarray,
                           train_ratio: float, val_ratio: float, seed: int = 42):
    """Split by measurement file while preserving class balance."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    split_ratios = np.array([train_ratio, val_ratio, max(0.0, 1.0 - train_ratio - val_ratio)])
    positive_splits = split_ratios > 0

    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        cls_groups = np.unique(groups[cls_indices])
        rng.shuffle(cls_groups)

        n_groups = len(cls_groups)
        if n_groups < int(positive_splits.sum()):
            raise ValueError(
                f"Class {cls} has {n_groups} files; cannot split by file with the requested ratios."
            )
        desired = split_ratios * n_groups
        counts = np.floor(desired).astype(int)
        remaining = n_groups - int(counts.sum())
        if remaining > 0:
            for split_idx in np.argsort(-(desired - counts))[:remaining]:
                counts[split_idx] += 1

        for split_idx in np.where(positive_splits & (counts == 0))[0]:
            donor_candidates = np.where(counts > 1)[0]
            if len(donor_candidates) == 0:
                raise ValueError(
                    f"Class {cls} has {n_groups} files; cannot split by file with the requested ratios."
                )
            donor = donor_candidates[np.argmax(counts[donor_candidates])]
            counts[donor] -= 1
            counts[split_idx] += 1

        n_train, n_val, _ = counts.tolist()

        train_groups = set(cls_groups[:n_train])
        val_groups = set(cls_groups[n_train:n_train + n_val])
        test_groups = set(cls_groups[n_train + n_val:])

        train_idx.extend([idx for idx in cls_indices if groups[idx] in train_groups])
        val_idx.extend([idx for idx in cls_indices if groups[idx] in val_groups])
        test_idx.extend([idx for idx in cls_indices if groups[idx] in test_groups])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def prepare_dataloaders(args):
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"data_dir not found: {args.data_dir}")

    use_few_shot = args.train_samples_per_class is not None
    if use_few_shot and args.train_samples_per_class <= 0:
        raise ValueError("train_samples_per_class 必须为正数")

    if not use_few_shot:
        total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            print(f"Warning: 划分比例之和为 {total_ratio}，将自动归一化")
            args.train_ratio /= total_ratio
            args.val_ratio /= total_ratio
            args.test_ratio /= total_ratio
        val_share = None
        test_share = None
    else:
        remaining_ratio = args.val_ratio + args.test_ratio
        if remaining_ratio <= 0:
            raise ValueError("使用小样本模式时，val_ratio 和 test_ratio 不能同时为 0")
        val_share = args.val_ratio / remaining_ratio
        test_share = args.test_ratio / remaining_ratio

    print("Preparing Datasets...")
    if use_few_shot:
        print(f"从 {args.data_dir} 加载数据，小样本模式：每类训练样本={args.train_samples_per_class}，"
              f"验证/测试按剩余数据的 {val_share:.2f}/{test_share:.2f} 划分")
        print("已设置 train_samples_per_class，train_ratio 将被忽略。")
    else:
        print(f"从 {args.data_dir} 加载数据，划分比例: 训练={args.train_ratio:.2f}, 验证={args.val_ratio:.2f}, 测试={args.test_ratio:.2f}")

    dataset_name = getattr(args, "dataset", "pu")
    if dataset_name == "pu":
        full_dataset = PUBS10Dataset(
            args.data_dir,
            window_size=args.window_size,
            stride=args.stride,
            condition=getattr(args, "pu_condition", PU_DEFAULT_CONDITIONS_TEXT),
            channel=getattr(args, "pu_channel", "vibration_1"),
            measurement_start=getattr(args, "pu_measurement_start", 1),
            measurement_end=getattr(args, "pu_measurement_end", 3),
        )
    elif dataset_name == "pu_a2r":
        full_dataset = None
    elif dataset_name == "cwru":
        full_dataset = CWRUDataset(args.data_dir, window_size=args.window_size, stride=args.stride)
        full_dataset.class_names = CWRU_CLASS_NAMES
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name == "pu_a2r":
        train_domain = getattr(args, "pu_train_domain", "artificial")
        test_domain = getattr(args, "pu_test_domain", "real")
        print(f"[PU-A2R] train_domain={train_domain}, test_domain={test_domain}")

        common_max_bearings = getattr(args, "pu_max_bearings_per_class", None)
        train_max_bearings = getattr(args, "pu_train_max_bearings_per_class", None)
        test_max_bearings = getattr(args, "pu_test_max_bearings_per_class", None)
        if train_max_bearings is None:
            train_max_bearings = common_max_bearings
        if test_max_bearings is None:
            test_max_bearings = common_max_bearings

        a2r_kwargs = dict(
            window_size=args.window_size,
            stride=args.stride,
            condition=getattr(args, "pu_condition", PU_DEFAULT_CONDITIONS_TEXT),
            channel=getattr(args, "pu_channel", "vibration_1"),
            measurement_start=getattr(args, "pu_measurement_start", 1),
            measurement_end=getattr(args, "pu_measurement_end", 3),
        )
        train_full = PUA2RDataset(
            args.data_dir,
            domain=train_domain,
            max_bearings_per_class=train_max_bearings,
            **a2r_kwargs,
        )
        args.class_names = list(train_full.class_names)

        if use_few_shot:
            raise ValueError("PU-A2R 暂不支持 few-shot 模式 (train_samples_per_class)")

        if train_domain == test_domain:
            print("PU-A2R 同域：在同一域上做文件级 train/val/test 划分")
            train_indices, val_indices, test_indices = stratified_group_split(
                np.asarray(train_full.labels),
                train_full.groups,
                args.train_ratio,
                args.val_ratio,
                seed=args.seed,
            )
            train_ds = Subset(train_full, train_indices)
            val_ds = Subset(train_full, val_indices)
            test_ds = Subset(train_full, test_indices)
        else:
            print("PU-A2R 跨域：训练域做 train/val 文件级划分，测试域整体作为测试集")
            test_full = PUA2RDataset(
                args.data_dir,
                domain=test_domain,
                max_bearings_per_class=test_max_bearings,
                **a2r_kwargs,
            )
            denom = args.train_ratio + args.val_ratio
            if denom <= 0:
                raise ValueError("train_ratio + val_ratio 必须大于 0 才能在 PU-A2R 跨域模式下划分训练/验证集")
            ratio_train = args.train_ratio / denom
            ratio_val = args.val_ratio / denom
            train_indices, val_indices, _ = stratified_group_split(
                np.asarray(train_full.labels),
                train_full.groups,
                ratio_train,
                ratio_val,
                seed=args.seed,
            )
            train_ds = Subset(train_full, train_indices)
            val_ds = Subset(train_full, val_indices)
            test_ds = test_full

        train_size = len(train_ds)
        val_size = len(val_ds)
        test_size = len(test_ds)
        total_size = train_size + val_size + test_size
    else:
        args.class_names = list(getattr(full_dataset, "class_names", CLASS_NAMES))
        total_size = len(full_dataset)
        labels = np.asarray(full_dataset.labels)

        def _norm_conditions(text):
            return frozenset(t.strip() for t in (text or "").split(",") if t.strip())

        test_condition = getattr(args, "pu_test_condition", None)
        cross_condition = (
            dataset_name == "pu"
            and test_condition
            and _norm_conditions(test_condition) != _norm_conditions(args.pu_condition)
        )
        use_file_split = (
            dataset_name == "pu"
            and not use_few_shot
            and not getattr(args, "no_file_split", False)
            and hasattr(full_dataset, "groups")
            and not cross_condition
        )

        if cross_condition:
            if use_few_shot:
                raise ValueError("PU 跨工况暂不支持 few-shot 模式 (train_samples_per_class)")
            train_m_start = getattr(args, "pu_measurement_start", 1)
            train_m_end = getattr(args, "pu_measurement_end", 3)
            test_m_start = getattr(args, "pu_test_measurement_start", None) or train_m_start
            test_m_end = getattr(args, "pu_test_measurement_end", None) or train_m_end
            print(
                f"PU 跨工况：训练={args.pu_condition} (meas {train_m_start}..{train_m_end})"
                f" → 测试={test_condition} (meas {test_m_start}..{test_m_end})"
            )
            test_full = PUBS10Dataset(
                args.data_dir,
                window_size=args.window_size,
                stride=args.stride,
                condition=test_condition,
                channel=getattr(args, "pu_channel", "vibration_1"),
                measurement_start=test_m_start,
                measurement_end=test_m_end,
            )
            denom = args.train_ratio + args.val_ratio
            if denom <= 0:
                raise ValueError("train_ratio + val_ratio 必须大于 0 才能在 PU 跨工况模式下划分训练/验证集")
            ratio_train = args.train_ratio / denom
            ratio_val = args.val_ratio / denom
            train_indices, val_indices, _ = stratified_group_split(
                labels,
                full_dataset.groups,
                ratio_train,
                ratio_val,
                seed=args.seed,
            )
            train_ds = Subset(full_dataset, train_indices)
            val_ds = Subset(full_dataset, val_indices)
            test_ds = test_full
            train_size = len(train_ds)
            val_size = len(val_ds)
            test_size = len(test_ds)
            total_size = train_size + val_size + test_size
        elif use_few_shot:
            train_indices, val_indices, test_indices = stratified_few_shot_split(
                labels,
                args.train_samples_per_class,
                args.val_ratio,
                args.test_ratio,
                seed=args.seed,
            )
            train_ds = Subset(full_dataset, train_indices)
            val_ds = Subset(full_dataset, val_indices)
            test_ds = Subset(full_dataset, test_indices)
            train_size, val_size, test_size = len(train_indices), len(val_indices), len(test_indices)
        elif use_file_split:
            print("PU 数据使用测量文件级划分，避免同一 .mat 文件的窗口同时出现在训练和测试中。")
            train_indices, val_indices, test_indices = stratified_group_split(
                labels,
                full_dataset.groups,
                args.train_ratio,
                args.val_ratio,
                seed=args.seed,
            )
            train_ds = Subset(full_dataset, train_indices)
            val_ds = Subset(full_dataset, val_indices)
            test_ds = Subset(full_dataset, test_indices)
            train_size, val_size, test_size = len(train_indices), len(val_indices), len(test_indices)
        else:
            train_size = int(total_size * args.train_ratio)
            val_size = int(total_size * args.val_ratio)
            test_size = total_size - train_size - val_size
            generator = torch.Generator().manual_seed(args.seed)
            train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)

    print(f"数据集划分: 训练集={train_size}, 验证集={val_size}, 测试集={test_size}, 总计={total_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)
    return train_loader, val_loader, test_loader


def evaluate_noise_robustness(model, loader, criterion, device, snr_list, noise_types):
    """Evaluate accuracy for each noise type x SNR combination.

    Returns a nested dict ``{noise_type: {"clean": acc, snr: acc, ...}, ...}``.
    The ``clean`` measurement is computed once (it is identical across types)
    and copied into every sub-dict so the plot helper can find it.
    """
    if isinstance(noise_types, str):
        noise_types = [noise_types]

    results = {}
    _, acc_clean = evaluate(model, loader, criterion, device, snr_db=None)
    print(f"  Clean (no noise): {acc_clean:.2f}%")

    for nt in noise_types:
        sub = {"clean": acc_clean}
        print(f"\n  --- Noise type: {nt} ---")
        for snr in snr_list:
            _, acc = evaluate(model, loader, criterion, device, snr_db=snr, noise_type=nt)
            sub[snr] = acc
            print(f"    [{nt}] SNR = {snr:>4} dB: {acc:.2f}%")
        results[nt] = sub
    return results


def plot_noise_robustness(results, title: str, save_path: str):
    """Plot one accuracy-vs-SNR curve per noise type, plus a clean baseline."""
    plt.figure(figsize=(10, 6))

    first_type = next(iter(results))
    first_sub = results[first_type]
    snr_values = sorted(k for k in first_sub if k != "clean")

    for nt, sub in results.items():
        accs = [sub[s] for s in snr_values]
        plt.plot(snr_values, accs, marker="o", linewidth=1.8, label=nt)

    plt.axhline(y=first_sub["clean"], color="r", linestyle="--",
                label=f'Clean: {first_sub["clean"]:.2f}%')
    plt.xlabel("SNR (dB)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xticks(snr_values)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Noise robustness plot saved to {save_path}")


def maybe_report_model_stats(model, window_size: int, device, thop_available: bool,
                             profile_fn=None, clever_format_fn=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if thop_available and profile_fn is not None and clever_format_fn is not None:
        dummy_input = torch.randn(1, 1, window_size).to(device)
        flops, params = profile_fn(model, inputs=(dummy_input,), verbose=False)
        flops_str, params_str = clever_format_fn([flops, params], "%.3f")
        print(f"FLOPs: {flops_str}")
        print(f"Params (thop): {params_str}")
    else:
        print("FLOPs: Not available (install thop: pip install thop)")


def report_periodic_ffn_gamma(model, threshold: float = 1e-4):
    """Print learned gamma statistics for PeriodicBottleneckFFN modules.

    ``gamma`` is zero-initialised in PeriodicBottleneckFFN. If it remains close
    to zero after training, the model mostly ignored the periodic residual. If
    it grows, the model learned to mix in the sinusoidal branch.
    """
    found = False
    for name, module in model.named_modules():
        gamma = getattr(module, "gamma", None)
        alpha = getattr(module, "alpha", None)
        phase = getattr(module, "phase", None)
        if gamma is None or alpha is None or phase is None:
            continue

        with torch.no_grad():
            g = gamma.detach().float().cpu()
            a = alpha.detach().float().cpu()
            p = phase.detach().float().cpu()
            abs_g = g.abs()
            active_ratio = (abs_g > threshold).float().mean().item() * 100.0
            preview = ", ".join(f"{v:.4g}" for v in g[:8].tolist())

        display_name = name if name else module.__class__.__name__
        print(
            f"{display_name} periodic gamma | "
            f"mean_abs={abs_g.mean().item():.6f}, "
            f"max_abs={abs_g.max().item():.6f}, "
            f"rms={torch.sqrt((g ** 2).mean()).item():.6f}, "
            f"active>{threshold:g}={active_ratio:.1f}%"
        )
        print(
            f"{display_name} periodic alpha/phase | "
            f"alpha_mean={a.mean().item():.6f}, alpha_std={a.std(unbiased=False).item():.6f}, "
            f"phase_mean={p.mean().item():.6f}, phase_std={p.std(unbiased=False).item():.6f}"
        )
        print(f"{display_name} gamma first8 = [{preview}]")
        found = True

    if found:
        print("Tip: gamma near 0 means the periodic residual stayed mostly off; larger |gamma| means it was used.")


def _plot_history(train_losses, val_losses, train_accs, val_accs, config: ExperimentConfig):
    loss_title = "Loss"
    acc_title = "Accuracy"
    if config.history_title_suffix:
        loss_title = f"Loss ({config.history_title_suffix})"
        acc_title = f"Accuracy ({config.history_title_suffix})"

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title(loss_title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title(acc_title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(config.history_plot_path)
    print(f"Training history saved to {config.history_plot_path}")


def _print_classification_report(y_true, y_pred, zero_division: Optional[int], class_names):
    kwargs = {
        "labels": list(range(len(class_names))),
        "target_names": class_names,
        "digits": 4,
    }
    if zero_division is not None:
        kwargs["zero_division"] = zero_division
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, **kwargs))


def _configure_moh_routing(model, hard_routing: bool, balance_enabled: bool) -> int:
    changed = 0
    for module in model.modules():
        if hasattr(module, "hard_routing"):
            module.hard_routing = bool(hard_routing)
            changed += 1
        if hasattr(module, "balance_loss_weight"):
            if not hasattr(module, "_original_balance_loss_weight"):
                module._original_balance_loss_weight = float(module.balance_loss_weight)
            module.balance_loss_weight = (
                module._original_balance_loss_weight if balance_enabled else 0.0
            )
    return changed


def run_experiment(args, model, device, config: ExperimentConfig):
    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return
    config = _with_run_output_paths(args, config)
    class_names = getattr(args, "class_names", CLASS_NAMES)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.label_smoothing > 0:
        print(f"[INFO] CrossEntropyLoss label_smoothing={args.label_smoothing}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    snr_range = (args.train_snr_min, args.train_snr_max)

    if args.train_noise:
        print(f"Training with noise: SNR=[{args.train_snr_min}, {args.train_snr_max}] dB, "
              f"per_sample={args.snr_per_sample}, type={args.noise_type}")
    if args.val_noise:
        print(f"Validation with noise: SNR=[{args.train_snr_min}, {args.train_snr_max}] dB, "
              f"per_sample={args.snr_per_sample}, type={args.noise_type}")

    warmup_epochs = int(getattr(args, "moh_soft_warmup_epochs", 0) or 0)
    if warmup_epochs > 0:
        print(f"MoH routing warm-up: soft routing for first {warmup_epochs} epoch(s); "
              "balance loss disabled during warm-up.")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        if warmup_epochs > 0:
            in_warmup = epoch <= warmup_epochs
            changed = _configure_moh_routing(
                model,
                hard_routing=not in_warmup,
                balance_enabled=not in_warmup,
            )
            if changed and (epoch == 1 or epoch == warmup_epochs + 1):
                mode = "soft routing" if in_warmup else "default routing"
                print(f"[MoH] Epoch {epoch:03d}: {mode}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            add_noise=args.train_noise, snr_range=snr_range,
            snr_per_sample=args.snr_per_sample, noise_type=args.noise_type,
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device,
            add_noise=args.val_noise, snr_range=snr_range,
            snr_per_sample=args.snr_per_sample, noise_type=args.noise_type,
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"train_acc {train_acc:.2f}% | val_loss {val_loss:.4f} | val_acc {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.best_model_path)

    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Saved best model to {config.best_model_path}")

    os.makedirs(getattr(args, "results_dir", "results"), exist_ok=True)
    csv_path = config.history_csv_path(args.train_snr_min, args.train_snr_max)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_loss", "train_acc", "val_loss", "val_acc"])
        for row in zip(train_losses, train_accs, val_losses, val_accs):
            writer.writerow([f"{row[0]:.4f}", f"{row[1]:.2f}", f"{row[2]:.4f}", f"{row[3]:.2f}"])
    print(f"Training history saved to {csv_path}")

    _plot_history(train_losses, val_losses, train_accs, val_accs, config)

    print("Evaluating on test set...")
    model.load_state_dict(torch.load(config.best_model_path, map_location=device))
    report_periodic_ffn_gamma(model)

    routing_stat_cases = []
    if args.test_noise:
        print(f"\n=== Noise Robustness Evaluation ({config.model_display_name}) ===")
        snr_list = [int(x) for x in args.snr_list.split(",")]
        test_noise_types = [t.strip() for t in args.test_noise_types.split(",") if t.strip()]
        if not test_noise_types:
            test_noise_types = ["gaussian"]
        invalid = [t for t in test_noise_types if t not in NOISE_TYPES]
        if invalid:
            raise ValueError(f"Unknown test noise types: {invalid}. Valid: {NOISE_TYPES}")
        print(f"  Sweeping noise types: {test_noise_types}")
        noise_results = evaluate_noise_robustness(
            model, test_loader, criterion, device,
            snr_list=snr_list, noise_types=test_noise_types,
        )
        plot_noise_robustness(noise_results, title=config.noise_plot_title, save_path=config.noise_plot_path)
        routing_stat_cases = [
            (f"test {nt} {snr} dB", snr, nt)
            for nt in test_noise_types
            for snr in snr_list
        ]

    report_moh_routing_stats(model, test_loader, device, "test clean")
    for label, snr, nt in routing_stat_cases:
        report_moh_routing_stats(model, test_loader, device, label, snr_db=snr, noise_type=nt)

    test_snr = args.test_snr
    if test_snr is not None:
        print(f"\nEvaluating with fixed SNR = {test_snr} dB (noise type={args.noise_type})")
    else:
        print("\nEvaluating on clean test set (no noise)")

    y_true, y_pred = evaluate_full(model, test_loader, device, snr_db=test_snr,
                                   noise_type=args.noise_type)
    _print_classification_report(y_true, y_pred, config.classification_zero_division, class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    title = config.confusion_title
    if test_snr is not None:
        title += f" (SNR={test_snr}dB)"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(config.confusion_matrix_path)
    print(f"Confusion matrix saved to {config.confusion_matrix_path}")

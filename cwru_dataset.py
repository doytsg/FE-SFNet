import os
import re

import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset


PU_BS_10_CLASSES = [
    ("K001", "Normal"),
    ("KA04", "Outer race"),
    ("KA15", "Outer race"),
    ("KA16", "Outer race"),
    ("KA22", "Outer race"),
    ("KA30", "Outer race"),
    ("KI16", "Inner race"),
    ("KI17", "Inner race"),
    ("KI18", "Inner race"),
    ("KI21", "Inner race"),
]

PU_BS_10_CLASS_NAMES = [
    f"{bearing_code} {fault_type}" for bearing_code, fault_type in PU_BS_10_CLASSES
]

PU_DEFAULT_CONDITIONS = ("N15_M01_F10", "N09_M07_F10", "N15_M07_F04")
PU_DEFAULT_CONDITIONS_TEXT = ",".join(PU_DEFAULT_CONDITIONS)


PU_3CLASS_NAMES = ["Healthy", "Outer race", "Inner race"]

PU_BEARING_GROUPS = {
    "artificial": {
        "Healthy": ["K002"],
        "Outer race": ["KA01", "KA05", "KA07"],
        "Inner race": ["KI01", "KI05", "KI07"],
    },
    "real": {
        "Healthy": ["K001"],
        "Outer race": ["KA04", "KA15", "KA16", "KA22", "KA30"],
        "Inner race": ["KI14", "KI16", "KI17", "KI18", "KI21"],
    },
}


def _matlab_string(value) -> str:
    arr = np.asarray(value).squeeze()
    if arr.shape == ():
        return str(arr.item())
    return str(arr)


def _parse_conditions(condition: str):
    if isinstance(condition, str):
        conditions = tuple(item.strip() for item in condition.split(",") if item.strip())
    else:
        conditions = tuple(condition)
    if not conditions:
        raise ValueError("At least one PU operating condition must be specified.")
    return conditions

class CWRUDataset(Dataset):
    """
    CWRU dataset loader (same preprocessing logic as the user's baseline script):
    - Walk a root directory and load all .mat files.
    - Read the signal from the key that endswith('DE_time').
    - Sliding-window segmentation into fixed-length samples.
    - Label mapping:
        0: Normal
        1-3: Ball (007,014,021)
        4-6: Inner (007,014,021)
        7-9: Outer (007,014,021)
    """
    def __init__(self, root_dir: str, window_size: int = 2048, stride: int = 2048):
        self.root_dir = root_dir
        self.window_size = int(window_size)
        self.stride = int(stride)

        self.data = []
        self.labels = []
        self._load_data()

        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)

    def _load_data(self):
        print(f"[CWRUDataset] Loading data from: {self.root_dir}")
        for root, _, files in os.walk(self.root_dir):
            for fn in files:
                if not fn.endswith(".mat"):
                    continue
                file_path = os.path.join(root, fn)

                label = self._get_label_from_path(file_path)
                if label is None:
                    continue

                signal = self._load_signal(file_path)
                if signal is None or len(signal) < self.window_size:
                    continue

                num_windows = (len(signal) - self.window_size) // self.stride + 1
                for i in range(num_windows):
                    start = i * self.stride
                    end = start + self.window_size
                    segment = signal[start:end]
                    self.data.append(segment)
                    self.labels.append(label)

        print(f"[CWRUDataset] Loaded {len(self.data)} samples from: {self.root_dir}")

    @staticmethod
    def _get_label_from_path(path: str):
        # normalize separators
        path = path.replace("\\", "/")

        if "Normal Baseline Data" in path:
            return 0

        is_ball = "Ball" in path
        is_inner = "Inner Race" in path
        is_outer = "Outer Race" in path
        if not (is_ball or is_inner or is_outer):
            return None

        if "0.007" in path:
            d_idx = 0
        elif "0.014" in path:
            d_idx = 1
        elif "0.021" in path:
            d_idx = 2
        else:
            return None

        if is_ball:
            base_offset = 1
        elif is_inner:
            base_offset = 4
        else:
            base_offset = 7

        return base_offset + d_idx

    @staticmethod
    def _load_signal(file_path: str):
        try:
            mat = scipy.io.loadmat(file_path)
            # Find key like 'XXX_DE_time'
            for key in mat.keys():
                if key.endswith("DE_time"):
                    return mat[key].flatten()
            return None
        except Exception as e:
            print(f"[CWRUDataset] Error reading {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (1, T)
        x = self.data[idx][None, :]
        y = self.labels[idx]
        x = torch.from_numpy(x)  # float32
        return x, y


class PUBS10Dataset(Dataset):
    """
    PU-BS-10 dataset loader using the ALMFormer-style 10-class setup and
    representative operating conditions:
        N15_M01_F10 = 1500 rpm, 0.1 Nm, 1000 N
        N09_M07_F10 = 900 rpm, 0.7 Nm, 1000 N
        N15_M07_F04 = 1500 rpm, 0.7 Nm, 400 N

    Class mapping:
        0: K001 Normal
        1: KA04 Outer race
        2: KA15 Outer race
        3: KA16 Outer race
        4: KA22 Outer race
        5: KA30 Outer race
        6: KI16 Inner race
        7: KI17 Inner race
        8: KI18 Inner race
        9: KI21 Inner race
    """

    FILE_RE = re.compile(
        r"^(?P<condition>N\d+_M\d+_F\d+)_(?P<bearing_code>K[A-Z0-9]+)_(?P<seq>\d+)\.mat$"
    )

    def __init__(
        self,
        root_dir: str,
        window_size: int = 2048,
        stride: int = 2048,
        condition: str = PU_DEFAULT_CONDITIONS_TEXT,
        channel: str = "vibration_1",
        measurement_start: int = 1,
        measurement_end: int = 3,
    ):
        self.root_dir = root_dir
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.conditions = _parse_conditions(condition)
        self.condition = ",".join(self.conditions)
        self.channel = channel
        self.measurement_start = int(measurement_start)
        self.measurement_end = int(measurement_end)
        self.class_names = PU_BS_10_CLASS_NAMES
        self.label_map = {
            bearing_code: label for label, (bearing_code, _) in enumerate(PU_BS_10_CLASSES)
        }

        self.data = []
        self.labels = []
        self.groups = []
        self.file_counts = {name: 0 for name in self.class_names}
        self.sample_counts = {name: 0 for name in self.class_names}
        self._load_data()

        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.groups = np.asarray(self.groups)

    def _load_data(self):
        print(f"[PUBS10Dataset] Loading data from: {self.root_dir}")
        print(f"[PUBS10Dataset] Conditions={self.condition}, channel={self.channel}")

        files = []
        for root, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                parsed = self._parse_filename(fn)
                if parsed is None:
                    continue
                condition, bearing_code, seq = parsed
                if condition not in self.conditions:
                    continue
                if bearing_code not in self.label_map:
                    continue
                if not (self.measurement_start <= seq <= self.measurement_end):
                    continue
                files.append((self.label_map[bearing_code], condition, seq, os.path.join(root, fn)))

        files.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        if not files:
            raise ValueError(
                f"No PU-BS-10 files found in {self.root_dir} for conditions {self.condition}."
            )

        for label, _, _, file_path in files:
            signal = self._load_signal(file_path, self.channel)
            if signal is None or len(signal) < self.window_size:
                continue

            class_name = self.class_names[label]
            self.file_counts[class_name] += 1
            num_windows = (len(signal) - self.window_size) // self.stride + 1
            for i in range(num_windows):
                start = i * self.stride
                end = start + self.window_size
                segment = signal[start:end]
                self.data.append(segment)
                self.labels.append(label)
                self.groups.append(file_path)
                self.sample_counts[class_name] += 1

        print(f"[PUBS10Dataset] Loaded {len(self.data)} samples from {len(files)} files")
        for idx, class_name in enumerate(self.class_names):
            print(
                f"  class {idx}: {class_name} | files={self.file_counts[class_name]} "
                f"| windows={self.sample_counts[class_name]}"
            )

    def _parse_filename(self, filename: str):
        match = self.FILE_RE.match(filename)
        if match is None:
            return None
        return (
            match.group("condition"),
            match.group("bearing_code"),
            int(match.group("seq")),
        )

    @staticmethod
    def _load_signal(file_path: str, channel: str):
        try:
            mat = scipy.io.loadmat(file_path)
            mat_keys = [key for key in mat.keys() if not key.startswith("__")]
            for key in mat_keys:
                root = mat[key]
                if not hasattr(root, "dtype") or root.dtype.names is None:
                    continue
                if "Y" not in root.dtype.names:
                    continue
                y_channels = root[0, 0]["Y"]
                for entry in y_channels.reshape(-1):
                    name = _matlab_string(entry["Name"])
                    if name == channel:
                        return np.asarray(entry["Data"]).reshape(-1)
            print(f"[PUBS10Dataset] Channel '{channel}' not found in {file_path}")
            return None
        except Exception as e:
            print(f"[PUBS10Dataset] Error reading {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][None, :]
        y = self.labels[idx]
        x = torch.from_numpy(x)
        return x, y


class PUA2RDataset(Dataset):
    """
    PU 3-class dataset for Artificial-vs-Real damage experiments.

    Class mapping:
        0: Healthy
        1: Outer race
        2: Inner race

    ``domain`` selects which bearing group to load:
        - "artificial": K002 / KA01,KA05,KA07 / KI01,KI05,KI07
        - "real":      K001 / KA04,KA15,KA16,KA22,KA30 / KI14,KI16,KI17,KI18,KI21
    """

    FILE_RE = re.compile(
        r"^(?P<condition>N\d+_M\d+_F\d+)_(?P<bearing_code>K[A-Z0-9]+)_(?P<seq>\d+)\.mat$"
    )

    def __init__(
        self,
        root_dir: str,
        window_size: int = 2048,
        stride: int = 2048,
        condition: str = PU_DEFAULT_CONDITIONS_TEXT,
        channel: str = "vibration_1",
        measurement_start: int = 1,
        measurement_end: int = 3,
        domain: str = "artificial",
        max_bearings_per_class: int | None = None,
    ):
        if domain not in PU_BEARING_GROUPS:
            raise ValueError(f"Unsupported PU domain: {domain}; expected one of {list(PU_BEARING_GROUPS)}")

        self.root_dir = root_dir
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.conditions = _parse_conditions(condition)
        self.condition = ",".join(self.conditions)
        self.channel = channel
        self.measurement_start = int(measurement_start)
        self.measurement_end = int(measurement_end)
        self.domain = domain
        self.max_bearings_per_class = max_bearings_per_class
        self.class_names = list(PU_3CLASS_NAMES)

        bearings_map = PU_BEARING_GROUPS[domain]
        self.label_map = {}
        self.selected_bearings = {}
        for label, class_name in enumerate(self.class_names):
            bearings = list(bearings_map[class_name])
            if max_bearings_per_class is not None and max_bearings_per_class > 0:
                bearings = bearings[:max_bearings_per_class]
            self.selected_bearings[class_name] = bearings
            for code in bearings:
                self.label_map[code] = label

        self.data = []
        self.labels = []
        self.groups = []
        self.file_counts = {name: 0 for name in self.class_names}
        self.sample_counts = {name: 0 for name in self.class_names}
        self._load_data()

        self.data = np.asarray(self.data, dtype=np.float32)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.groups = np.asarray(self.groups)

    def _load_data(self):
        print(f"[PUA2RDataset] Loading domain={self.domain} from {self.root_dir}")
        print(f"[PUA2RDataset] Conditions={self.condition}, channel={self.channel}")
        if self.max_bearings_per_class is not None:
            print(f"[PUA2RDataset] max_bearings_per_class={self.max_bearings_per_class}; "
                  f"selected={self.selected_bearings}")

        files = []
        for root, _, filenames in os.walk(self.root_dir):
            for fn in filenames:
                match = self.FILE_RE.match(fn)
                if match is None:
                    continue
                condition = match.group("condition")
                bearing_code = match.group("bearing_code")
                seq = int(match.group("seq"))
                if condition not in self.conditions:
                    continue
                if bearing_code not in self.label_map:
                    continue
                if not (self.measurement_start <= seq <= self.measurement_end):
                    continue
                files.append((self.label_map[bearing_code], condition, seq, os.path.join(root, fn)))

        files.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        if not files:
            raise ValueError(
                f"No PU files found in {self.root_dir} for domain={self.domain} conditions={self.condition}."
            )

        for label, _, _, file_path in files:
            signal = PUBS10Dataset._load_signal(file_path, self.channel)
            if signal is None or len(signal) < self.window_size:
                continue

            class_name = self.class_names[label]
            self.file_counts[class_name] += 1
            num_windows = (len(signal) - self.window_size) // self.stride + 1
            for i in range(num_windows):
                start = i * self.stride
                end = start + self.window_size
                segment = signal[start:end]
                self.data.append(segment)
                self.labels.append(label)
                self.groups.append(file_path)
                self.sample_counts[class_name] += 1

        print(f"[PUA2RDataset] Loaded {len(self.data)} samples from {len(files)} files")
        for idx, class_name in enumerate(self.class_names):
            print(
                f"  class {idx}: {class_name} | files={self.file_counts[class_name]} "
                f"| windows={self.sample_counts[class_name]}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][None, :]
        y = self.labels[idx]
        x = torch.from_numpy(x)
        return x, y

# SGSFNet

Code for **SGSFNet: Subband-Guided Spectral Filtering Network** for lightweight, noise-robust bearing fault diagnosis from raw one-dimensional vibration signals.

This repository contains the training code, model definitions, dataset loaders, and baseline implementations used in the experiments. Datasets, trained checkpoints, generated figures, logs, and result files are intentionally not included.

## Highlights

- Raw 1-D vibration signal classification for PU and CWRU bearing datasets.
- SGSFNet implementation with a robust local front end and attention-free spectral token mixing.
- Unified training entry for SGSFNet and several comparison models.
- SNR-controlled synthetic noise augmentation and evaluation.
- Few-shot training support through fixed samples per class.
- File-level splitting for PU data to reduce leakage between training and testing.

## Repository Structure

```text
.
|-- train_model.py                  # Unified training entry
|-- train_common.py                 # Shared training, evaluation, splitting, and noise utilities
|-- cwru_dataset.py                 # CWRU, PU-BS-10, and PU-A2R dataset loaders
|-- export_tsne_features.py         # Optional feature export and t-SNE utility
|-- models/
|   |-- sds_dsfb_transformer.py     # SGSFNet implementation, kept with legacy file name
|   |-- sds_frontend.py             # Robust feature stem and Haar-Down modules
|   |-- refined_dsfb_modules.py     # Spectral Filter Mixer and Li-FFN modules
|   |-- cnn_transformer.py
|   |-- convformer_nse.py
|   |-- liconvformer.py
|   |-- almformer.py
|   |-- wdcnn.py
|   |-- tslanet.py
|   |-- drsn_cw.py
|   `-- gtfenet.py
`-- run_*.bat                       # Optional Windows experiment launch scripts
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For Linux/macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Data Preparation

The datasets are not distributed with this repository. Download the original CWRU and PU bearing datasets from their official sources, then place or symlink them locally.

Default data roots used by the scripts:

```text
PU_extracted/    # PU .mat files
data/            # CWRU .mat files
```

For PU-BS-10, the loader expects `.mat` files whose names follow this pattern:

```text
N15_M01_F10_K001_1.mat
N09_M07_F10_KA04_2.mat
...
```

For CWRU, the loader recursively scans `.mat` files and uses directory names such as `Normal Baseline Data`, `Ball`, `Inner Race`, and `Outer Race` to infer labels.

You can also pass a custom data directory:

```bash
python train_model.py sgsfnet --data_dir path/to/PU_extracted
python train_model.py sgsfnet --dataset cwru --data_dir path/to/CWRU
```

## Quick Start

Train SGSFNet on the default PU-BS-10 setup:

```bash
python train_model.py sgsfnet --data_dir PU_extracted --epochs 100
```

Train under fixed Gaussian noise and evaluate multiple SNR levels:

```bash
python train_model.py sgsfnet ^
  --data_dir PU_extracted ^
  --train_noise --val_noise --snr_per_sample ^
  --noise_type gaussian --train_snr_min -12 --train_snr_max -12 ^
  --test_noise --test_noise_types gaussian ^
  --snr_list -12,-8,-4,0 ^
  --epochs 100
```

Few-shot training:

```bash
python train_model.py sgsfnet --data_dir PU_extracted --train_samples_per_class 50
```

CWRU training:

```bash
python train_model.py sgsfnet --dataset cwru --data_dir data --num_classes 10
```

The historical command name is still supported:

```bash
python train_model.py sds_dsfb ...
```

## Supported Models

The unified entry supports:

- `sgsfnet` / `sds_dsfb`
- `cnn_transformer`
- `convformer_nse`
- `liconvformer`
- `almformer`
- `wdcnn`
- `tslanet`
- `drsn_cw`
- `gtfenet`
- `mslk`

Show command-line options with:

```bash
python train_model.py -h
python train_model.py sgsfnet -h
```

## Outputs

Training writes checkpoints, curves, confusion matrices, and CSV logs to the configured results directory, usually `results/`. These generated files are ignored by git and should not be committed.

Useful options:

- `--results_dir results`
- `--run_name my_run`
- `--seed 42`
- `--num_workers 0`
- `--batch_size 128`
- `--window_size 2048`
- `--stride 2048`

## Notes

- The SGSFNet implementation is kept in files with the earlier internal name `sds_dsfb` for backward compatibility with existing scripts.
- No pretrained weights are included. Train your own checkpoints from the downloaded datasets.
- Reported numbers can vary with GPU, PyTorch version, split settings, seeds, and dataset preprocessing.

## Citation

If this repository is useful for your research, please cite the corresponding SGSFNet paper when it becomes available.

## License

This project is released under the MIT License.

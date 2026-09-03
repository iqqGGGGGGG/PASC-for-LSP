# PASC

**P**olarity-**A**symmetric **S**tructural **C**alibration for Link Sign Prediction.

**[English](README.md)** | **[中文](README_zh.md)**

Two model variants are provided:

| Variant | Description |
|---------|-------------|
| **PASC-H** | Hard twin supervision with residual-contrastive learning |
| **PASC-S** | Soft residual-aware reweighting |

## Project Structure

```
PASC/
├── PASC-H/
│   ├── train.py              # Training & evaluation entry point
│   ├── models.py             # PASC_H model, RGA, CondDecoder, etc.
│   ├── struct_utils.py       # Community detection (SSSNET), twin matching, structural gradient
│   ├── load_data.py          # Data loading & IID/OOD splitting
│   ├── utils.py              # Feature generation (SVD / Laplacian), logger
│   ├── _config-*.yaml        # Per-dataset hyperparameter configs
│   ├── Data/
│   │   └── data_file/        # Place signed edge lists here (not tracked)
│   └── logs/                 # Training logs
├── PASC-S/
│   └── (same structure as PASC-H)
└── README.md
```

## Environment Setup

```bash
conda create -n PASC python=3.10 -y
conda activate PASC

# PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Dependencies
pip install torch_geometric torch_geometric_signed_directed \
    numpy scipy scikit-learn pyyaml networkx

# Or install the non-PyTorch dependencies from the lockable list
pip install -r requirements.txt
```

## Datasets

Dataset files are not distributed with this repository. Download the original
datasets from SNAP, prepare each file as a three-column signed edge list, and
place it under `Data/data_file/` of the variant you want to run:

| Dataset | File |
|---------|------|
| Bitcoin-Alpha | `Bitcoin-Alpha.txt` |
| Bitcoin-OTC | `Bitcoin-OTC.txt` |
| Epinions | `Epinions.txt` |
| Slashdot | `Slashdot.txt` |
| Wikipedia-RfA | `wkr.txt` |

Each line follows the format: `source_node target_node sign` (sign: `1` for positive, `-1` for negative).

The expected format is `source_node target_node sign`, where the sign is `1`
for a positive edge and `-1` for a negative edge. See
[DATA_NOTICE.md](DATA_NOTICE.md) for source links and licensing notes.

## Usage

### Data Preprocessing (first run)

The first training run will automatically:
1. Extract the largest connected component (LCC)
2. Remap node IDs to `[0, N)`
3. Split into train / val / test via MST-based strategy
4. Generate signed spectral features (Laplacian or SVD)
5. Run SSSNET community detection & structural twin matching
6. Cache results under `Data/cache/`

### Training

```bash
# PASC-H on Bitcoin-OTC
cd PASC-H
python train.py --config _config-OTC.yaml

# PASC-S on Bitcoin-Alpha
cd ../PASC-S
python train.py --config _config-Alpha.yaml

# Multiple runs for mean +/- std reporting
python train.py --config _config-OTC.yaml --num_runs 5
```

### Key Arguments

Values from the selected YAML file override the fallback values shown below.

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `_config-OTC.yaml` | YAML config file path |
| `--dataset` | `Bitcoin-OTC` | Dataset name |
| `--datapath` | `Data/` | Directory containing `data_file/` and generated `cache/` |
| `--gpu` | `3` | GPU device ID (`-1` for CPU) |
| `--epochs` | `500` | Max training epochs |
| `--batch_size` | `32768` | Training batch size |
| `--patience` | `40` | Early stopping patience |
| `--lr` | `0.0001` | Learning rate |
| `--dim_h` | `128` | Hidden dimension |
| `--dim_z` | `128` | Latent dimension |
| `--dropout` | `0.2` | Dropout rate |
| `--l2reg` | `0.03` | L2 regularization |
| `--drop_edge_rate` | `0.2` | Edge dropout rate for augmentation |
| `--svd_rank` | `128` | SVD / Laplacian embedding rank |
| `--feature_type` | `laplacian` | Node feature type: `svd` or `laplacian` |
| `--num_runs` | `1` | Number of repeated runs |

### Evaluation Metrics

Reported at test time: **AUC**, **F1 (binary / macro / micro)**, **AP**, **Balanced Accuracy**, **Precision**, **Recall**.

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.4.1+cu121 |
| torch_geometric | 2.8.0 |
| torch_geometric_signed_directed | 1.1.1 |
| numpy | >= 1.24 |
| scipy | >= 1.10 |
| scikit-learn | >= 1.2 |
| pyyaml | >= 6.0 |
| networkx | >= 3.0 |

## Citation

```bibtex
@inproceedings{gao2026pasc,
    title={PASC: Polarity-Asymmetric Structural Calibration for Link Sign Prediction},
    author={Gao, Qiqi and Song, Wenzhuo and Liu, Xueyan},
    booktitle = {IEEE International Conference on Data Mining (ICDM)},
    year={2026}
}
```

# PASC

**P**olarity-**A**symmetric **S**tructural **C**alibration for Link Sign Prediction.

**[English](README.md)** | **[中文](README_zh.md)**

提供两种模型变体：

| 变体 | 说明 |
|------|------|
| **PASC-H** | 硬孪生监督 + 残差对比学习 |
| **PASC-S** | 软残差感知重加权 |

## 项目结构

```
PASC/
├── PASC-H/
│   ├── train.py              # 训练与评估入口
│   ├── models.py             # PASC_H 模型、RGA、CondDecoder 等
│   ├── struct_utils.py       # 社区检测 (SSSNET)、孪生匹配、结构梯度
│   ├── load_data.py          # 数据加载与 IID/OOD 切分
│   ├── utils.py              # 特征生成 (SVD / Laplacian)、日志工具
│   ├── _config-*.yaml        # 各数据集超参数配置
│   ├── Data/
│   │   └── data_file/        # 在此放置符号边列表（不纳入 Git）
│   └── logs/                 # 训练日志
├── PASC-S/
│   └── (结构同 PASC-H)
└── README.md
```

## 环境配置

```bash
conda create -n PASC python=3.10 -y
conda activate PASC

# PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 依赖包
pip install torch_geometric torch_geometric_signed_directed \
    numpy scipy scikit-learn pyyaml networkx

# 或从依赖清单安装 PyTorch 以外的依赖
pip install -r requirements.txt
```

## 数据集

本仓库不分发数据集文件。请从 SNAP 下载原始数据，将其整理为三列符号边列表，
再放入待运行变体的 `Data/data_file/` 目录：

| 数据集 | 文件 |
|--------|------|
| Bitcoin-Alpha | `Bitcoin-Alpha.txt` |
| Bitcoin-OTC | `Bitcoin-OTC.txt` |
| Epinions | `Epinions.txt` |
| Slashdot | `Slashdot.txt` |
| Wikipedia-RfA | `wkr.txt` |

每行格式：`源节点 目标节点 符号`（符号：`1` 表示正边，`-1` 表示负边）。
来源链接与许可说明见 [DATA_NOTICE.md](DATA_NOTICE.md)。

## 使用方法

### 数据预处理（首次运行）

首次训练时会自动完成以下步骤：
1. 提取最大连通分量 (LCC)
2. 将节点 ID 重映射为 `[0, N)`
3. 基于 MST 策略切分训练集 / 验证集 / 测试集
4. 生成符号谱特征（Laplacian 或 SVD）
5. 运行 SSSNET 社区检测与结构孪生匹配
6. 缓存结果至 `Data/cache/`

### 训练

```bash
# PASC-H 在 Bitcoin-OTC 上训练
cd PASC-H
python train.py --config _config-OTC.yaml

# PASC-S 在 Bitcoin-Alpha 上训练
cd ../PASC-S
python train.py --config _config-Alpha.yaml

# 多次运行以报告 mean +/- std
python train.py --config _config-OTC.yaml --num_runs 5
```

### 主要参数

所选 YAML 配置中的值会覆盖下表所列的回退值。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `_config-OTC.yaml` | YAML 配置文件路径 |
| `--dataset` | `Bitcoin-OTC` | 数据集名称 |
| `--datapath` | `Data/` | 包含 `data_file/` 与生成的 `cache/` 的目录 |
| `--gpu` | `3` | GPU 设备编号（`-1` 表示使用 CPU） |
| `--epochs` | `500` | 最大训练轮数 |
| `--batch_size` | `32768` | 训练批大小 |
| `--patience` | `40` | 早停耐心值 |
| `--lr` | `0.0001` | 学习率 |
| `--dim_h` | `128` | 隐藏层维度 |
| `--dim_z` | `128` | 潜在空间维度 |
| `--dropout` | `0.2` | Dropout 比率 |
| `--l2reg` | `0.03` | L2 正则化系数 |
| `--drop_edge_rate` | `0.2` | 边丢弃率（数据增强） |
| `--svd_rank` | `128` | SVD / Laplacian 嵌入秩 |
| `--feature_type` | `laplacian` | 节点特征类型：`svd` 或 `laplacian` |
| `--num_runs` | `1` | 重复运行次数 |

### 评估指标

测试阶段报告：**AUC**、**F1 (binary / macro / micro)**、**AP**、**Balanced Accuracy**、**Precision**、**Recall**。

## 依赖

| 包 | 版本 |
|----|------|
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

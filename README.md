# HiNet3D

HiNet3D is a medication recommendation framework that combines hierarchical patient-history modeling with 3D-aware drug representation learning from electronic health records (EHRs).

## Highlights

- Models each clinical visit as a node in a patient-history graph.
- Uses cross-visit pairwise aggregation to capture longitudinal clinical context.
- Encodes 3D molecular structures with a Geometric Vector Perceptron (GVP).
- Evaluates medication recommendation using Jaccard, PR-AUC, F1, and ROC-AUC.

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA-enabled GPU recommended
- RDKit
- PyTorch Geometric
- pandas, numpy, scikit-learn, tqdm, pyyaml

## Installation

```bash
git clone <your-repository-url>
cd HiNet3D

conda create -n hinet3d python=3.8 -y
conda activate hinet3d

pip install torch torchvision torchaudio
pip install torch-geometric rdkit pandas numpy scikit-learn tqdm pyyaml
```

> Install the PyTorch and PyTorch Geometric versions that match your CUDA version.

## Project Structure

```text
HiNet3D/
├── data/
│   ├── raw/                 # Original EHR and molecular files
│   ├── processed/           # Preprocessed patient and drug graphs
│   └── splits/              # Train/validation/test splits
├── configs/
│   └── default.yaml         # Experiment configuration
├── models/
│   ├── patient_encoder.py   # Patient-history graph encoder
│   ├── drug_encoder.py      # 3D molecular GVP encoder
│   └── hinet3d.py           # Complete HiNet3D model
├── scripts/
│   ├── preprocess.py        # Data preprocessing
│   ├── train.py             # Model training
│   └── evaluate.py          # Model evaluation
├── requirements.txt
└── README.md
```

## Data Preparation

Place the required files under `data/raw/`:

```text
data/raw/
├── diagnoses.csv
├── procedures.csv
├── medications.csv
├── patients.csv
└── molecules/               # 3D molecular structures, e.g. SDF/MOL files
```

The EHR data should contain patient identifiers, visit identifiers, diagnosis codes, procedure codes, and medication labels. Patient identifiers must be anonymized before use.

Preprocess the data:

```bash
python scripts/preprocess.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --config configs/default.yaml
```

## Configuration

Example `configs/default.yaml`:

```yaml
seed: 42
device: cuda

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.00001
  early_stopping_patience: 10

model:
  hidden_dim: 128
  patient_layers: 2
  drug_layers: 3
  dropout: 0.2
  use_3d: true

paths:
  data_dir: data/processed
  checkpoint_dir: checkpoints
  log_dir: logs
```

## Training

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data_dir data/processed \
  --output_dir checkpoints
```

Resume training from a checkpoint:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pt \
  --data_dir data/processed \
  --split test
```

The evaluation script reports:

- Jaccard
- PR-AUC
- F1-score
- ROC-AUC

## Reproducibility

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --seed 42 \
  --deterministic
```

For comparable results, use the same dataset version, preprocessing rules, train/validation/test split, random seed, and software versions reported in the paper.

## Troubleshooting

**CUDA out of memory**

```yaml
training:
  batch_size: 8
```

You may also reduce `hidden_dim` or enable gradient accumulation.

**No 3D molecular files found**

Check the file names, molecular identifiers, and the mapping between drug codes and molecular structures. The default configuration expects `use_3d: true`.

**Different CUDA or PyTorch versions**

Reinstall PyTorch Geometric after installing the correct PyTorch and CUDA versions.

## Disclaimer

This repository is intended for research purposes only. HiNet3D is not a medical device and must not be used to make clinical decisions without appropriate validation and professional oversight.

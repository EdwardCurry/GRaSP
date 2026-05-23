# GRaSP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20353446.svg)](https://doi.org/10.5281/zenodo.20353446)

This repository contains the code for the paper:

**"Mechanism-Driven Cross-Modality Modeling for Gene Regulatory Network Inference from Single-Cell Multi-Omics"**

GRaSP is a **mechanism-driven** framework for gene regulatory network (GRN) inference from paired single-cell RNA-seq and single-cell ATAC-seq data. It explicitly models the **transcription factor–regulatory element–target gene (TF–RE–TG) cascade** and learns regulatory associations under biologically feasible cross-modality interactions derived from TF binding motifs and genomic proximity priors.

---

## Repository Structure

```
.
├── preprocessing.py              # preprocessing utilities for RNA/ATAC matrices
├── data_augmentation.py          # train/test split and training-set augmentation
├── main.py                       # model training
├── interpret.py                  # Integrated Gradients-based GRN extraction
├── examples/
│   └── test_data/                # small test dataset for smoke testing
├── requirements.txt
├── LICENSE
└── README.md
```

Core scripts:

- **`preprocessing.py`**
  - Data preprocessing utilities.
  - Produces aligned RNA/ATAC matrices and cell type labels, or equivalent matrices that match the expected input format below.

- **`data_augmentation.py`**
    1. Splits raw cells into train/test sets before augmentation.
    2. Augments only the training set using metacell generation and within-cell-type shuffling.
    3. Keeps the test set as unaugmented raw cells.
    4. Saves train/test matrices for TFs, target genes, regulatory elements, and cell type labels.

- **`main.py`** — training only
  - Trains the joint GRaSP model with knowledge-guided sparse cross-attention.
  - Saves model checkpoints, fitted scalers, feature names, and run configuration.

- **`interpret.py`** — interpretability and GRN extraction only
  - Loads trained weights and fitted scalers.
  - Builds population-level or cell-type-specific profiles.
  - Runs Integrated Gradients (IG).
  - Outputs top-ranked TF–TG, TF–RE, and RE–TG edge lists.

---

## Installation

GRaSP is implemented in Python and PyTorch. We recommend using a clean conda environment.

```bash
conda create -n grasp python=3.10 -y
conda activate grasp
```

Install PyTorch following the command appropriate for your CUDA version from the [official PyTorch installation page](https://pytorch.org/get-started/locally/). For example, for a recent CUDA-enabled environment:

```bash
pip install torch torchvision torchaudio
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install numpy pandas scikit-learn transformers joblib captum linformer wandb
```

Notes:
- `captum` is required for `interpret.py`.
- `linformer` is optional. If unavailable, training falls back to a dense Transformer encoder, but may be slower for larger inputs.
- `wandb` is optional and can be disabled if experiment tracking is not needed.

---

## Inputs / Expected Files

### 1. Train/test matrices

By default, `main.py` expects the following CSV files under `--data_dir`:

```
train_tf_expression.csv
train_tg_expression.csv
train_atac_data.csv
train_cell_types.csv
test_tf_expression.csv
test_tg_expression.csv
test_atac_data.csv
test_cell_types.csv
```

All expression/accessibility CSVs are stored as **features × samples** and are loaded internally as **samples × features**.

Expected matrix meaning:
- `*_tf_expression.csv`: expression matrix for transcription factors.
- `*_tg_expression.csv`: expression matrix for target genes.
- `*_atac_data.csv`: chromatin accessibility matrix for regulatory elements/peaks.
- `*_cell_types.csv`: one cell type label per sample, aligned with the sample order of the matrices.

### 2. Knowledge priors

#### TF–RE motif prior

Preferred sparse format:
```
tf_re_motif_sparse.npz
```
with fields:
- `row` — integer row indices.
- `col` — integer column indices.
- `shape` — matrix shape `(num_RE, num_TF)`.

Fallback dense format:
```
tf_re_motif.csv
```
as a binary matrix with rows = RE/peak names and columns = TF names.

#### TG–RE TSS proximity prior

Preferred sparse format:
```
tg_re_sparse_matrix.npz
```
with fields:
- `row` — integer row indices.
- `col` — integer column indices.
- `shape` — matrix shape `(num_TG, num_RE)`.

> If priors are missing, the model may run with empty priors, but this is not recommended for meaningful knowledge-guided sparse attention.

---

## Example / Test Data

A small synthetic test dataset is provided under `examples/test_data/`. This dataset is intended only for **installation verification and smoke testing** — it is deliberately small and should not be used to reproduce paper-level benchmark results.

```
examples/test_data/
├── train_tf_expression.csv
├── train_tg_expression.csv
├── train_atac_data.csv
├── train_cell_types.csv
├── test_tf_expression.csv
├── test_tg_expression.csv
├── test_atac_data.csv
├── test_cell_types.csv
├── tf_re_motif_sparse.npz
└── tg_re_sparse_matrix.npz
```

---

## Quick Start: Reproducing Key Results with Test Data

The following commands run a minimal end-to-end test using the provided example dataset.

### Step 1. Train GRaSP on the example data

```bash
python main.py \
  --data_dir examples/test_data \
  --out_dir runs/grasp_example \
  --epochs 2 \
  --batch_size 16 \
  --lr 1e-4 \
  --alpha 0.5 \
  --lambda_align 0.5 \
  --max_neighbors 32 \
  --neg_k 8
```

Expected outputs:
```
runs/grasp_example/best_model.pt
runs/grasp_example/scaler_tf.joblib
runs/grasp_example/scaler_re.joblib
runs/grasp_example/scaler_tg.joblib
runs/grasp_example/feature_names.json
runs/grasp_example/run_config.json
```

### Step 2. Identify available test cell types

```bash
python -c "
import pandas as pd
labels = pd.read_csv('examples/test_data/test_cell_types.csv')
print(labels.iloc[:, 0].dropna().astype(str).unique())
"
```

Use one of the printed labels as `<CELL_TYPE>` in the next step.

### Step 3. Run Integrated Gradients and extract GRN edge lists

```bash
python interpret.py \
  --data_dir examples/test_data \
  --run_dir runs/grasp_example \
  --ckpt best_model.pt \
  --cell_type <CELL_TYPE> \
  --use_original_only \
  --n_steps 16 \
  --topk 50
```

Expected output directory:
```
runs/grasp_example/interpret/<CELL_TYPE>/
```

Expected output files:
```
IG_TF_to_TG_topk.csv
IG_RE_to_TG_topk.csv
IG_RE_to_TF_topk.csv
IG_TG_to_TF_topk.csv
```

### Step 4. Verify outputs

```bash
test -f runs/grasp_example/best_model.pt && echo "Model checkpoint found."
find runs/grasp_example/interpret -name "*topk.csv" -print
```

If the checkpoint and at least one `*topk.csv` file are present, the installation and example workflow are functioning correctly.

---

## Running GRaSP on a Full Dataset

### Step 1. Prepare data

If starting from raw paired scRNA-seq and scATAC-seq data, run the preprocessing and augmentation pipeline. The output should match the files listed in **Inputs / Expected Files**.

```bash
python data_augmentation.py
```

### Step 2. Train the model

```bash
python main.py \
  --data_dir /path/to/processed_data \
  --out_dir runs/grasp_full \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-4 \
  --alpha 0.5 \
  --lambda_align 0.5 \
  --max_neighbors 256 \
  --neg_k 64
```

### Step 3. Interpret a cell type

```bash
python interpret.py \
  --data_dir /path/to/processed_data \
  --run_dir runs/grasp_full \
  --ckpt best_model.pt \
  --cell_type <CELL_TYPE> \
  --use_original_only \
  --n_steps 50 \
  --topk 200
```

Optional: interpret only a subset of targets to reduce runtime:

```bash
python interpret.py \
  --data_dir /path/to/processed_data \
  --run_dir runs/grasp_full \
  --ckpt best_model.pt \
  --cell_type <CELL_TYPE> \
  --tg_targets 0:200 \
  --tf_targets 0:100 \
  --topk 200
```

---

## Public Data Sources Used in the Paper

The full benchmark datasets are not required for the smoke test above. The paper-level experiments use public datasets and resources:

- **PBMC dataset**: 10x Genomics 10k Human PBMCs, Multiome v1.0, Chromium X.
- **BMMC dataset**: NeurIPS 2021 Single-Cell Competition dataset.
- **TF–RE benchmark labels**: CistromeDB ChIP-seq resources.
- **TF–TG benchmark labels**: KnockTF KO/KD resources.
- **RE–TG benchmark labels**: GTEx eQTL resources.
- **TF motif priors**: JASPAR 2024.

Users who want to reproduce the full benchmark should download the above public datasets, run the preprocessing pipeline, construct motif/TSS priors, and then train/evaluate GRaSP using the full-data commands described above.

---

## Reproducibility Notes

- Raw cells should be split into training, validation, and test sets **before** data augmentation.
- Augmentation should be applied only to the training set.
- Scalers should be fit on the training set only and reused for validation/test/interpretability inference.
- Reported evaluation in the manuscript uses threshold-free ranking metrics such as AUROC and AUPRC/AUPR ratio.
- The example/test dataset is for workflow verification only; it is not intended to reproduce manuscript-level performance.

---

## Troubleshooting

### `captum` import error

```bash
pip install captum
```

### CUDA out-of-memory error

Reduce one or more of the following arguments:
```
--batch_size
--max_neighbors
--neg_k
--topk
```

### No Integrated Gradients output for a cell type

Check that the requested cell type appears in `test_cell_types.csv`:
```bash
python -c "
import pandas as pd
labels = pd.read_csv('examples/test_data/test_cell_types.csv')
print(labels.iloc[:, 0].dropna().astype(str).unique())
"
```

### Empty or uninformative regulatory edges

Confirm that the prior matrices are present and non-empty:
```bash
ls -lh examples/test_data/*motif* examples/test_data/*sparse*
```

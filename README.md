# GRaSP

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20353446.svg)](https://doi.org/10.5281/zenodo.20353446)

This repository contains the code for the paper:

**“Mechanism-Driven Cross-Modality Modeling for Gene Regulatory Network Inference from Single-Cell Multi-Omics”**

GRaSP is a **mechanism-driven** framework for gene regulatory network (GRN) inference from paired single-cell RNA-seq and single-cell ATAC-seq data. It explicitly models the **transcription factor–regulatory element–target gene (TF–RE–TG) cascade** and learns regulatory associations under biologically feasible cross-modality interactions derived from TF binding motifs and genomic proximity priors.


## Repository Structure

```text
.
├── preprocessing.py              # preprocessing utilities for RNA/ATAC matrices
├── data_augmentation.py          # train/test split and training-set augmentation
├── main.py                       # model training
├── interpret.py                  # Integrated Gradients-based GRN extraction
├── examples/
│   └── test_data/                # small reviewer/test dataset for smoke testing
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

Install PyTorch following the command appropriate for your CUDA version from the official PyTorch installation page. For example, for a recent CUDA-enabled environment:

```bash
pip install torch torchvision torchaudio
```

Install the remaining dependencies:

```bash
pip install numpy pandas scikit-learn transformers joblib captum linformer wandb
```

Notes:

- `captum` is required for `interpret.py`.
- `linformer` is optional. If it is unavailable, training can fall back to a dense Transformer encoder, but this may be slower for larger inputs.
- `wandb` is optional and can be disabled if experiment tracking is not needed.

---

## Inputs / Expected Files

### 1. Train/test matrices

By default, `main.py` expects the following CSV files under `--data_dir`:

```text
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

```text
tf_re_motif_sparse.npz
```

with fields:

- `row` — integer row indices.
- `col` — integer column indices.
- `shape` — matrix shape `(num_RE, num_TF)`.

Fallback dense format:

```text
tf_re_motif.csv
```

as a binary matrix with rows = RE/peak names and columns = TF names.

#### TG–RE TSS proximity prior

Preferred sparse format:

```text
tg_re_sparse_matrix.npz
```

with fields:

- `row` — integer row indices.
- `col` — integer column indices.
- `shape` — matrix shape `(num_TG, num_RE)`.

If priors are missing, the model may run with empty priors, but this is not recommended for meaningful knowledge-guided sparse attention.

---

## Example / Test Data for Reviewers

A small test dataset is provided under:

```text
examples/test_data/
```

This dataset is intended only for **installation checking and smoke testing**. It is deliberately small and should not be used to reproduce the paper-level benchmark results.

The directory should contain the same files expected by `main.py`:

```text
examples/test_data/
├── train_tf_expression.csv
├── train_tg_expression.csv
├── train_atac_data.csv
├── train_cell_types.csv
├── test_tf_expression.csv
├── test_tg_expression.csv
├── test_atac_data.csv
├── test_cell_types.csv
├── tf_re_motif_sparse.npz          # or tf_re_motif.csv
└── tg_re_sparse_matrix.npz
```

The example dataset is designed to test the following functions:

1. loading TF, TG, and RE matrices;
2. loading motif and TSS-proximity priors;
3. fitting scalers on the training set;
4. training a small GRaSP model;
5. saving checkpoints and configuration files;
6. running Integrated Gradients on at least one cell type;
7. generating top-ranked regulatory edge lists.

---

## Quick Start: Run the Example/Test Data

The following commands run a minimal end-to-end test using the small example dataset.

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

```text
runs/grasp_example/best_model.pt
runs/grasp_example/scaler_tf.joblib
runs/grasp_example/scaler_re.joblib
runs/grasp_example/scaler_tg.joblib
runs/grasp_example/feature_names.json
runs/grasp_example/run_config.json
```

### Step 2. Identify an available test cell type

If you are not sure which cell type label is present in the example dataset, run:

```bash
python - <<'PY'
import pandas as pd
labels = pd.read_csv('examples/test_data/test_cell_types.csv')
print(labels.iloc[:, 0].dropna().astype(str).unique())
PY
```

Use one of the printed labels as `<CELL_TYPE_NAME>` in the next command.

### Step 3. Run Integrated Gradients and extract GRN edge lists

```bash
python interpret.py \
  --data_dir examples/test_data \
  --run_dir runs/grasp_example \
  --ckpt best_model.pt \
  --cell_type <CELL_TYPE_NAME> \
  --use_original_only \
  --n_steps 16 \
  --topk 50
```

Expected output directory:

```text
runs/grasp_example/interpret/<CELL_TYPE_NAME>/
```

Expected output files include:

```text
IG_TF_to_TG_topk.csv
IG_RE_to_TG_topk.csv
IG_RE_to_TF_topk.csv
IG_TG_to_TF_topk.csv
```

### Step 4. Optional output check

```bash
test -f runs/grasp_example/best_model.pt && echo "Model checkpoint found."
find runs/grasp_example/interpret -name "*topk.csv" -print
```

If the checkpoint and at least one `*topk.csv` file are present, the installation and example workflow are functioning.

---

## Running GRaSP on a Full Dataset

### Step 1. Prepare data

If starting from raw paired scRNA-seq and scATAC-seq data, first run the preprocessing and data augmentation pipeline. The exact preprocessing command depends on the input format. The output should match the files listed in **Inputs / Expected Files**.

Example:

```bash
python data_augmentation.py
```

This should generate the required `train_*.csv` and `test_*.csv` files in the selected output directory.

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

Training outputs will be saved to `runs/grasp_full/`.

### Step 3. Interpret a cell type

```bash
python interpret.py \
  --data_dir /path/to/processed_data \
  --run_dir runs/grasp_full \
  --ckpt best_model.pt \
  --cell_type <CELL_TYPE_NAME> \
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
  --cell_type <CELL_TYPE_NAME> \
  --tg_targets 0:200 \
  --tf_targets 0:100 \
  --topk 200
```

---

## Public Data Sources Used in the Paper

The full benchmark datasets are not required for the smoke test above. The paper-level experiments use public datasets and public benchmark resources:

- PBMC dataset: 10x Genomics 10k Human PBMCs, Multiome v1.0, Chromium X.
- BMMC dataset: NeurIPS 2021 Single-Cell Competition dataset.
- TF–RE benchmark labels: CistromeDB ChIP-seq resources.
- TF–TG benchmark labels: KnockTF KO/KD resources.
- RE–TG benchmark labels: GTEx eQTL resources.
- TF motif priors: JASPAR 2024.

Users who want to reproduce the full benchmark should download the above public datasets, run the preprocessing pipeline, construct motif/TSS priors, and then train/evaluate GRaSP using the full-data commands described above.

---

## Reproducibility Notes

- Raw cells should be split into training, validation, and test sets **before** data augmentation.
- Augmentation should be applied only to the training set.
- Scalers should be fit on the training set only and reused for validation/test/interpretable inference.
- Reported evaluation in the manuscript uses threshold-free ranking metrics such as AUROC and AUPRC/AUPR ratio.
- The example/test dataset is for workflow verification only; it is not intended to reproduce manuscript-level performance.

---

## Troubleshooting

### `captum` import error

Install Captum:

```bash
pip install captum
```

### CUDA out-of-memory error

Reduce one or more of the following:

```bash
--batch_size
--max_neighbors
--neg_k
--topk
```

### No Integrated Gradients output for a cell type

Check that the requested cell type appears in `test_cell_types.csv`:

```bash
python - <<'PY'
import pandas as pd
labels = pd.read_csv('examples/test_data/test_cell_types.csv')
print(labels.iloc[:, 0].dropna().astype(str).unique())
PY
```

### Empty or uninformative regulatory edges

Confirm that the prior matrices are present and non-empty:

```bash
ls -lh examples/test_data/*motif* examples/test_data/*sparse* examples/test_data/*tss* 2>/dev/null
```

---

## License

Please specify the repository license here, for example MIT, BSD-3-Clause, Apache-2.0, or GPL-compatible license.

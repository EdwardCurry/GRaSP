# GRaSP

This repository contains the code used in the paper:

**"Inferring Gene Regulatory Network from Single-cell Multi-omics Data with Knowledge Guided Sparse Attention"**

The current codebase implements a **sparse, prior-guided neighbor attention**:
- Cross-attention is computed **only on prior-allowed edges** (no dense TF×RE / RE×TG attention matrix).
- The **alignment loss is strictly consistent** with this sparse implementation:
  - positives = allowed prior edges
  - negatives = sampled from disallowed edges with collision avoidance
- Integrated Gradients (IG) interpretability is separated into a dedicated script (`interpret.py`).

---

## Repository Structure

- **`preprocessing.py`**
  - Data preprocessing utilities.
  - Produces aligned RNA/ATAC matrices and cell type labels (or equivalent).

- **`data_augmentation.py`**
    1) Split into train/test first  
    2) Augment only the training set (metacells + within-cell-type shuffling)  
    3) Keep test as pure raw cells  
    4) Save train/test matrices for TF / TG / RE and cell type labels  

- **`main.py`**  (TRAINING ONLY)
  - Trains the joint model with **sparse neighbor attention** guided by knowledge priors.
  - Saves:
    - `best_model.pt` (and optionally `final_model.pt`)
    - `scaler_tf.joblib`, `scaler_re.joblib`, `scaler_tg.joblib` (fit on train only)
    - `feature_names.json`
    - `run_config.json`

- **`interpret.py`**  (INTERPRETABILITY ONLY)
  - Loads trained weights + scalers.
  - Builds a **cell-type-specific metacell** and runs Integrated Gradients (IG).
  - Outputs **TopK edge lists** (recommended for scalability) and optional dense attribution matrices.

---

## Inputs / Expected Files

### 1) Augmented datasets (produced by `data_augmentation.py`)
By default, `main.py` expects these CSV files under `--data_dir`:

- `train_tf_expression.csv`
- `train_tg_expression.csv`
- `train_atac_data.csv`
- `train_cell_types.csv`
- `test_tf_expression.csv`
- `test_tg_expression.csv`
- `test_atac_data.csv`
- `test_cell_types.csv`

All expression/accessibility CSVs are stored as **(features × samples)** and loaded as **(samples × features)** internally.

### 2) Knowledge priors (optional but recommended)

#### TF-RE motif prior (RE × TF)
Preferred sparse format:
- `tf_re_motif_sparse.npz` with fields:
  - `row` (int64), `col` (int64), `shape` = (num_RE, num_TF)

Fallback dense format:
- `tf_re_motif.csv` as a binary matrix with rows=RE names, columns=TF names.

#### TG-RE TSS prior (TG × RE)
Sparse format:
- `tg_re_sparse_matrix.npz` with fields:
  - `row` (int64), `col` (int64), `shape` = (num_TG, num_RE)

> If priors are missing, the model will run with **empty priors** (no allowed edges),
> which is not recommended for meaningful “knowledge-guided sparse attention”.

---

## Installation

Recommended Python packages:
- `torch`
- `numpy`, `pandas`
- `scikit-learn`
- `transformers`
- `joblib`
- `captum` (for interpretability)
- `linformer` (optional; if not installed, training falls back to a dense TransformerEncoder)
- `wandb` (optional)

---

## Usage

### Step 1: Data Augmentation
Run `data_augmentation.py` to generate train/test split and augmented training set.

Example:
```bash
python data_augmentation.py
```

This will create `train_*.csv` and `test_*.csv` files in the working directory (or the directory used by your script).

---

### Step 2: Train the Model (Training Only)

Train with sparse neighbor attention:

```bash
python main.py \
  --data_dir . \
  --out_dir runs/grasp_sparse_joint \
  --epochs 100 \
  --batch_size 256 \
  --lr 1e-4 \
  --alpha 0.5 \
  --lambda_align 0.5 \
  --max_neighbors 256 \
  --neg_k 64
```

Outputs will be saved to:

* `runs/grasp_sparse_joint/best_model.pt`
* `runs/grasp_sparse_joint/scaler_tf.joblib`, `scaler_re.joblib`, `scaler_tg.joblib`
* `runs/grasp_sparse_joint/feature_names.json`
* `runs/grasp_sparse_joint/run_config.json`

---

### Step 3: Interpretability (Integrated Gradients) in a Separate Script

Run IG on a specific cell type metacell:

```bash
python interpret.py \
  --data_dir . \
  --run_dir runs/grasp_sparse_joint \
  --ckpt best_model.pt \
  --cell_type Astrocytes \
  --use_original_only \
  --n_steps 50 \
  --topk 200
```

Optional: interpret only a subset of targets to reduce runtime:

```bash
python interpret.py \
  --data_dir . \
  --run_dir runs/grasp_sparse_joint \
  --cell_type Astrocytes \
  --tg_targets 0:200 \
  --tf_targets 0:100 \
  --topk 200
```

IG outputs will be saved under:

* `runs/grasp_sparse_joint/interpret/<cell_type>/`

Recommended outputs:

* `IG_TF_to_TG_topk.csv`
* `IG_RE_to_TG_topk.csv`
* `IG_RE_to_TF_topk.csv`
* `IG_TG_to_TF_topk.csv`

---

## Notes on Scalability

* The cross-attention implementation is **sparse** and only computes attention on prior-allowed edges.
* The alignment loss does **not** use dense BCE over all pairs; instead it uses:

  * positives = allowed edges
  * negatives = sampled disallowed edges with collision avoidance
* For interpretability, long-format TopK edge lists are preferred to avoid generating huge dense matrices.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRaSP (Training Only) - True Sparse Neighbor Attention
======================================================
This script trains the joint model with:
  1) True sparse cross-attention restricted to prior-allowed edges only.
  2) Alignment loss that is strictly consistent with the sparse implementation:
       - Positive edges: allowed neighbors from prior
       - Negative edges: sampled from disallowed set with collision-avoidance
  3) Joint objective: L_total = alpha * L1 + (1-alpha) * L2

This training script does NOT run Integrated Gradients (IG).
Interpretability is moved to interpret.py for:
  - loading weights + scalers + metacell per cell-type
  - running IG without blocking training and reducing OOM risk
"""

import os
import json
import math
import argparse
import random
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
import joblib

from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Optional: wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

# Optional: Linformer (if unavailable, fallback to dense TransformerEncoder as a compatibility fallback)
try:
    from linformer import Linformer
    LINFORMER_AVAILABLE = True
except Exception:
    LINFORMER_AVAILABLE = False


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    # General
    seed: int = 42

    # Optimization
    lr: float = 1e-4
    epochs: int = 100
    batch_size: int = 256
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Model
    embedding_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    dim_feedforward: int = 2048
    activation: str = "gelu"

    # Encoder (Linformer)
    num_layers: int = 1
    k_linformer: int = 256

    # Joint objective
    alpha: float = 0.5
    lambda_align: float = 0.5

    # Sparse attention controls
    max_neighbors: int = 256         # Cap degree per query if prior is too dense
    neg_k: int = 64                  # Negative samples per query for alignment loss
    neg_resample_steps: int = 3      # Resample steps to avoid collisions with allowed edges

    # Logging
    use_wandb: bool = False
    wandb_project: str = "GRaSP_SparseJoint"


# -------------------------
# Data I/O
# -------------------------
def load_expression_matrix(path: str) -> pd.DataFrame:
    """
    Read CSV stored as (features x samples) and transpose into (samples x features).
    """
    return pd.read_csv(path, index_col=0).T


def load_data(data_dir: str):
    """
    Load train/test CSVs produced by data_augmentation.py.

    Expected files in data_dir:
      - train_tf_expression.csv
      - train_tg_expression.csv
      - train_atac_data.csv
      - test_tf_expression.csv
      - test_tg_expression.csv
      - test_atac_data.csv
    """
    train_tf = load_expression_matrix(os.path.join(data_dir, "train_tf_expression.csv"))
    train_tg = load_expression_matrix(os.path.join(data_dir, "train_tg_expression.csv"))
    train_re = load_expression_matrix(os.path.join(data_dir, "train_atac_data.csv"))

    test_tf = load_expression_matrix(os.path.join(data_dir, "test_tf_expression.csv"))
    test_tg = load_expression_matrix(os.path.join(data_dir, "test_tg_expression.csv"))
    test_re = load_expression_matrix(os.path.join(data_dir, "test_atac_data.csv"))

    tf_names = train_tf.columns.tolist()
    tg_names = train_tg.columns.tolist()
    re_names = train_re.columns.tolist()

    # Sanity checks (feature order must match between train/test)
    assert tf_names == test_tf.columns.tolist(), "TF features mismatch between train and test."
    assert tg_names == test_tg.columns.tolist(), "TG features mismatch between train and test."
    assert re_names == test_re.columns.tolist(), "RE features mismatch between train and test."

    return (train_tf, train_re, train_tg), (test_tf, test_re, test_tg), (tf_names, re_names, tg_names)


# -------------------------
# Scaling (fit on train only)
# -------------------------
def fit_and_transform_scalers(
    train_tf: pd.DataFrame,
    train_re: pd.DataFrame,
    train_tg: pd.DataFrame,
    test_tf: pd.DataFrame,
    test_re: pd.DataFrame,
    test_tg: pd.DataFrame,
):
    """
    Fit scalers only on training set, then transform both train and test.
    Return scaled arrays and fitted scaler objects.
    """
    scaler_tf = StandardScaler()
    scaler_re = StandardScaler()
    scaler_tg = StandardScaler()

    train_tf_s = scaler_tf.fit_transform(train_tf.values)
    test_tf_s = scaler_tf.transform(test_tf.values)

    train_re_s = scaler_re.fit_transform(train_re.values)
    test_re_s = scaler_re.transform(test_re.values)

    train_tg_s = scaler_tg.fit_transform(train_tg.values)
    test_tg_s = scaler_tg.transform(test_tg.values)

    scalers = {"tf": scaler_tf, "re": scaler_re, "tg": scaler_tg}
    return (train_tf_s, train_re_s, train_tg_s), (test_tf_s, test_re_s, test_tg_s), scalers


# -------------------------
# Dataset
# -------------------------
class JointDataset(Dataset):
    """Joint dataset that returns (TF, RE, TG) for each sample."""
    def __init__(self, tf_data, re_data, tg_data):
        self.tf = torch.tensor(tf_data, dtype=torch.float32)
        self.re = torch.tensor(re_data, dtype=torch.float32)
        self.tg = torch.tensor(tg_data, dtype=torch.float32)
        assert self.tf.shape[0] == self.re.shape[0] == self.tg.shape[0], "Sample count mismatch."

    def __len__(self):
        return self.tf.shape[0]

    def __getitem__(self, idx):
        return self.tf[idx], self.re[idx], self.tg[idx]


# -------------------------
# Priors -> COO edges
# -------------------------
def load_tf_re_motif_edges(data_dir: str, re_names, tf_names):
    """
    Load TF-RE motif prior. Expected orientation: RE x TF (rows=RE, cols=TF).

    Preferred sparse format:
      tf_re_motif_sparse.npz with fields:
        - row (int64)
        - col (int64)
        - shape (tuple) == (num_RE, num_TF)

    Fallback dense format:
      tf_re_motif.csv with rows=RE names, cols=TF names, entries in {0,1}.

    Returns:
      rows_re, cols_tf, shape
    """
    npz = os.path.join(data_dir, "tf_re_motif_sparse.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        rows = d["row"].astype(np.int64)
        cols = d["col"].astype(np.int64)
        shape = tuple(d["shape"])
        if shape != (len(re_names), len(tf_names)):
            raise ValueError(f"Motif npz shape {shape} != (RE={len(re_names)}, TF={len(tf_names)})")
        return rows, cols, shape

    csv = os.path.join(data_dir, "tf_re_motif.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0)
        df = df.reindex(index=re_names, columns=tf_names, fill_value=0)
        mat = df.values
        rows, cols = np.nonzero(mat)
        return rows.astype(np.int64), cols.astype(np.int64), (len(re_names), len(tf_names))

    print("Warning: motif prior not found. Using empty prior (no allowed edges).")
    return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), (len(re_names), len(tf_names))


def load_tg_re_tss_edges(data_dir: str, tg_names, re_names):
    """
    Load TG-RE TSS prior. Expected orientation: TG x RE (rows=TG, cols=RE).

    Sparse format:
      tg_re_sparse_matrix.npz with fields:
        - row (int64)
        - col (int64)
        - shape (tuple) == (num_TG, num_RE)

    Returns:
      rows_tg, cols_re, shape
    """
    npz = os.path.join(data_dir, "tg_re_sparse_matrix.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        rows = d["row"].astype(np.int64)
        cols = d["col"].astype(np.int64)
        shape = tuple(d["shape"])
        if shape != (len(tg_names), len(re_names)):
            raise ValueError(f"TSS npz shape {shape} != (TG={len(tg_names)}, RE={len(re_names)})")
        return rows, cols, shape

    print("Warning: TSS prior not found. Using empty prior (no allowed edges).")
    return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), (len(tg_names), len(re_names))


# -------------------------
# Neighbor tables from COO edges
# -------------------------
def build_neighbor_table_from_coo(
    num_q: int,
    num_k: int,
    rows: np.ndarray,
    cols: np.ndarray,
    max_neighbors: Optional[int],
    seed: int,
    fallback_empty: str = "none",  # "none" | "random1"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build padded neighbor table for sparse attention from COO edges (query=row -> key=col).

    Returns:
      nbr_idx:   [num_q, Dmax] int64 (padded with 0)
      nbr_valid: [num_q, Dmax] bool

    Notes:
      - This avoids building any dense adjacency matrix.
      - If a query has no allowed neighbors and fallback_empty="none", its row will be all invalid.
    """
    assert rows.shape == cols.shape
    rng = np.random.default_rng(seed)

    if rows.size == 0:
        # No edges at all
        nbr_idx = torch.zeros((num_q, 1), dtype=torch.long)
        nbr_valid = torch.zeros((num_q, 1), dtype=torch.bool)
        if fallback_empty == "random1":
            nbr_idx[:, 0] = torch.randint(0, num_k, (num_q,), dtype=torch.long)
            nbr_valid[:, 0] = True
        return nbr_idx, nbr_valid

    # Sort by row to create CSR-like pointers
    order = np.argsort(rows, kind="mergesort")
    rows_s = rows[order]
    cols_s = cols[order]

    counts = np.bincount(rows_s, minlength=num_q)
    indptr = np.zeros(num_q + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    degs = np.zeros(num_q, dtype=np.int64)
    for q in range(num_q):
        start, end = indptr[q], indptr[q + 1]
        d = end - start
        if max_neighbors is not None and d > max_neighbors:
            d = max_neighbors
        degs[q] = d

    Dmax = int(degs.max()) if degs.size > 0 else 0
    if Dmax == 0:
        nbr_idx = torch.zeros((num_q, 1), dtype=torch.long)
        nbr_valid = torch.zeros((num_q, 1), dtype=torch.bool)
        if fallback_empty == "random1":
            nbr_idx[:, 0] = torch.randint(0, num_k, (num_q,), dtype=torch.long)
            nbr_valid[:, 0] = True
        return nbr_idx, nbr_valid

    nbr_idx = np.zeros((num_q, Dmax), dtype=np.int64)
    nbr_valid = np.zeros((num_q, Dmax), dtype=np.bool_)

    for q in range(num_q):
        start, end = indptr[q], indptr[q + 1]
        neigh = cols_s[start:end]
        if neigh.size == 0:
            if fallback_empty == "random1":
                nbr_idx[q, 0] = rng.integers(0, num_k)
                nbr_valid[q, 0] = True
            continue

        if max_neighbors is not None and neigh.size > max_neighbors:
            pick = rng.choice(neigh.size, size=max_neighbors, replace=False)
            neigh = neigh[pick]

        neigh = np.sort(neigh)
        d = neigh.size
        nbr_idx[q, :d] = neigh
        nbr_valid[q, :d] = True

    return torch.from_numpy(nbr_idx).long(), torch.from_numpy(nbr_valid).bool()


# -------------------------
# Encoders
# -------------------------
class FallbackTransformerEncoder(nn.Module):
    """Fallback dense TransformerEncoder if Linformer is not installed."""
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.enc(x)


def make_encoder(d_model: int, seq_len: int, cfg: Config) -> nn.Module:
    """
    Create an encoder module. Prefer Linformer if available, otherwise fallback.
    """
    if LINFORMER_AVAILABLE:
        k = min(cfg.k_linformer, seq_len)
        return Linformer(dim=d_model, seq_len=seq_len, depth=cfg.num_layers, heads=cfg.num_heads, k=k)
    return FallbackTransformerEncoder(d_model, cfg.num_heads, cfg.num_layers, cfg.dropout)


# -------------------------
# True Sparse Neighbor Attention
# -------------------------
class SparseNeighborAttention(nn.Module):
    """
    True sparse attention: compute attention ONLY on allowed neighbor keys.
    Alignment loss is strictly consistent with this sparse implementation:
      - positives: allowed edges
      - negatives: sampled disallowed edges with collision avoidance
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        nbr_idx: torch.Tensor,     # [Nq, Dmax] long
        nbr_valid: torch.Tensor,   # [Nq, Dmax] bool
        key_len: int,              # Nk
        neg_k: int,
        neg_resample_steps: int,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.drop = nn.Dropout(dropout)

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.neg_k = int(neg_k)
        self.neg_resample_steps = int(neg_resample_steps)
        self.key_len = int(key_len)

        # Register buffers
        self.register_buffer("nbr_idx", nbr_idx.long())       # [Nq, Dmax]
        self.register_buffer("nbr_valid", nbr_valid.bool())   # [Nq, Dmax]

        # Prepare sorted allowed list per query for collision checking using searchsorted
        sentinel = self.key_len  # padding sentinel (never sampled)
        idx_filled = torch.where(self.nbr_valid, self.nbr_idx, torch.full_like(self.nbr_idx, sentinel))
        idx_sorted, _ = torch.sort(idx_filled, dim=1)
        self.register_buffer("allowed_sorted", idx_sorted.long())  # [Nq, Dmax], padded with sentinel

    def _gather_by_index(self, x_all: torch.Tensor, idx_qd: torch.Tensor) -> torch.Tensor:
        """
        Efficient gather:
          x_all: [B, Nk, D]
          idx_qd: [Nq, D]  (2D index per query)
        Returns:
          [B, Nq, D, Dmodel]
        """
        B, Nk, Dm = x_all.shape
        idx_safe = idx_qd.to(x_all.device).clamp(0, Nk - 1)
        # Advanced indexing: [B, Nq, D, Dm]
        return x_all[:, idx_safe, :]

    def _collision_mask(self, neg_idx: torch.Tensor) -> torch.Tensor:
        """
        Check if neg_idx is in allowed set for each query using batched searchsorted.
          neg_idx: [Nq, K]
        Returns:
          hit: [Nq, K] bool
        """
        allowed = self.allowed_sorted  # [Nq, Dmax] sorted, padding=sentinel
        Nq, Dmax = allowed.shape
        pos = torch.searchsorted(allowed, neg_idx, right=False)  # [Nq, K]
        pos_clamp = pos.clamp(min=0, max=Dmax - 1)
        gathered = torch.gather(allowed, dim=1, index=pos_clamp)
        hit = (pos < Dmax) & (gathered == neg_idx)
        return hit

    def _sample_neg_indices(self, device: torch.device, seed: int) -> Optional[torch.Tensor]:
        """Sample negative key indices per query with collision avoidance."""
        if self.neg_k <= 0:
            return None

        Nq, _ = self.nbr_idx.shape
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        neg = torch.randint(0, self.key_len, (Nq, self.neg_k), generator=g, device=device)

        for _ in range(self.neg_resample_steps):
            hit = self._collision_mask(neg)
            if not hit.any():
                break
            num = int(hit.sum().item())
            neg[hit] = torch.randint(0, self.key_len, (num,), generator=g, device=device)

        return neg

    def forward(self, q: torch.Tensor, kv: torch.Tensor, compute_align: bool, seed: int):
        """
        q:  [B, Nq, D]
        kv: [B, Nk, D]
        Returns:
          out: [B, Nq, D]
          align_loss: scalar tensor (or None)
        """
        B, Nq, Dm = q.shape
        _, Nk, _ = kv.shape
        assert Nk == self.key_len, f"Key length mismatch: Nk={Nk} vs key_len={self.key_len}"

        device = q.device

        # Linear projections
        q_proj = self.Wq(q).view(B, Nq, self.num_heads, self.head_dim)  # [B,Nq,H,Dh]
        k_all = self.Wk(kv)                                             # [B,Nk,D]
        v_all = self.Wv(kv)                                             # [B,Nk,D]

        # Gather allowed neighbor keys/values
        k_g = self._gather_by_index(k_all, self.nbr_idx)  # [B,Nq,Dmax,D]
        v_g = self._gather_by_index(v_all, self.nbr_idx)

        # Reshape for multi-head
        Dmax = k_g.shape[2]
        k_g = k_g.view(B, Nq, Dmax, self.num_heads, self.head_dim)      # [B,Nq,Dmax,H,Dh]
        v_g = v_g.view(B, Nq, Dmax, self.num_heads, self.head_dim)

        # Compute logits only on neighbors
        logits = (q_proj.unsqueeze(2) * k_g).sum(dim=-1) * self.scale    # [B,Nq,Dmax,H]

        valid = self.nbr_valid.to(device).unsqueeze(0).unsqueeze(-1)     # [1,Nq,Dmax,1]
        logits = logits.masked_fill(~valid, float("-inf"))

        # Stabilize rows with no valid neighbors to avoid softmax(all -inf) -> NaN
        no_valid_row = ~valid.any(dim=2, keepdim=True)                   # [1,Nq,1,1]
        logits = logits.masked_fill(no_valid_row, 0.0)

        # Neighbor softmax
        attn = torch.softmax(logits, dim=2)                              # along Dmax
        attn = attn * valid.float()
        attn = self.drop(attn)

        # Weighted sum of V
        out = (attn.unsqueeze(-1) * v_g).sum(dim=2)                      # [B,Nq,H,Dh]
        out = out.reshape(B, Nq, Dm)
        out = self.Wo(out)

        align_loss = None
        if compute_align:
            # Only compute alignment for queries that have at least one valid neighbor
            row_has = self.nbr_valid.any(dim=1).to(device)               # [Nq]
            pos_logits_mean = logits.mean(dim=-1)                        # [B,Nq,Dmax]
            pos_valid = self.nbr_valid.to(device).unsqueeze(0) & row_has.unsqueeze(0).unsqueeze(-1)

            pos_logits = pos_logits_mean[pos_valid]
            if pos_logits.numel() == 0:
                loss_pos = torch.tensor(0.0, device=device)
            else:
                loss_pos = F.binary_cross_entropy_with_logits(
                    pos_logits, torch.ones_like(pos_logits), reduction="mean"
                )

            neg_idx = self._sample_neg_indices(device=device, seed=seed)
            if neg_idx is None:
                loss_neg = torch.tensor(0.0, device=device)
            else:
                k_neg = self._gather_by_index(k_all, neg_idx)            # [B,Nq,neg_k,D]
                k_neg = k_neg.view(B, Nq, self.neg_k, self.num_heads, self.head_dim)
                logits_neg = (q_proj.unsqueeze(2) * k_neg).sum(dim=-1) * self.scale  # [B,Nq,neg_k,H]
                logits_neg = logits_neg.mean(dim=-1)                      # [B,Nq,neg_k]

                logits_neg = logits_neg[:, row_has, :].reshape(-1)
                if logits_neg.numel() == 0:
                    loss_neg = torch.tensor(0.0, device=device)
                else:
                    loss_neg = F.binary_cross_entropy_with_logits(
                        logits_neg, torch.zeros_like(logits_neg), reduction="mean"
                    )

            align_loss = loss_pos + loss_neg

        return out, align_loss


class SparseCrossAttnFFNBlock(nn.Module):
    """A cross-attention block: sparse neighbor attention + FFN with residual/LayerNorm."""
    def __init__(self, attn: SparseNeighborAttention, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.attn = attn
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(attn.embed_dim)

        self.ff1 = nn.Linear(attn.embed_dim, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, attn.embed_dim)
        self.drop2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(attn.embed_dim)

        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, q, kv, compute_align: bool, seed: int):
        attn_out, align_loss = self.attn(q, kv, compute_align=compute_align, seed=seed)
        x = self.norm1(q + self.drop(attn_out))
        ff = self.ff2(self.drop2(self.act(self.ff1(x))))
        x = self.norm2(x + self.drop2(ff))
        return x, align_loss


# -------------------------
# Two flow models
# -------------------------
class TFREModel(nn.Module):
    """
    Flow 1:
      Inputs: TF (expression) + RE (accessibility)
      Output: TG prediction

    Cross-attn:
      TF attends to RE (TF->RE)
      RE attends to TF (RE->TF)
    Alignment loss = 0.5 * (align_tf2re + align_re2tf)
    """
    def __init__(
        self,
        num_TF: int,
        num_RE: int,
        num_TG: int,
        cfg: Config,
        shared_re: nn.ModuleDict,
        tf2re_block: SparseCrossAttnFFNBlock,
        re2tf_block: SparseCrossAttnFFNBlock
    ):
        super().__init__()
        self.num_TF = num_TF
        self.num_RE = num_RE
        self.num_TG = num_TG
        self.cfg = cfg
        d = cfg.embedding_dim

        # TF embeddings
        self.tf_id = nn.Embedding(num_TF, d)
        self.tf_val = nn.Sequential(nn.Linear(1, d), nn.ReLU())
        self.tf_merge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())

        # Shared RE embeddings
        self.re_id = shared_re["id"]
        self.re_val = shared_re["val"]
        self.re_merge = shared_re["merge"]

        # Self encoders
        self.enc_tf1 = make_encoder(d, num_TF, cfg)
        self.enc_re1 = make_encoder(d, num_RE, cfg)
        self.enc_tf2 = make_encoder(d, num_TF, cfg)
        self.enc_re2 = make_encoder(d, num_RE, cfg)

        # Sparse cross blocks
        self.cross_tf = tf2re_block
        self.cross_re = re2tf_block

        # Combine encoder (optional)
        self.combine = make_encoder(d, num_TF + num_RE, cfg)

        # Output head: map (TF+RE tokens) -> TG
        self.out1 = nn.Linear(num_TF + num_RE, num_TG)
        self.out2 = nn.Linear(d, 1)

    def forward(self, tf_in: torch.Tensor, re_in: torch.Tensor, seed: int):
        B = tf_in.size(0)
        device = tf_in.device

        tf_idx = torch.arange(self.num_TF, device=device).unsqueeze(0).expand(B, -1)
        re_idx = torch.arange(self.num_RE, device=device).unsqueeze(0).expand(B, -1)

        tf_emb = self.tf_merge(torch.cat([self.tf_id(tf_idx), self.tf_val(tf_in.unsqueeze(-1))], dim=-1))
        re_emb = self.re_merge(torch.cat([self.re_id(re_idx), self.re_val(re_in.unsqueeze(-1))], dim=-1))

        tf_h = self.enc_tf1(tf_emb)
        re_h = self.enc_re1(re_emb)

        tf_h, a1 = self.cross_tf(tf_h, re_h, compute_align=True, seed=seed)
        re_h, a2 = self.cross_re(re_h, tf_h, compute_align=True, seed=seed + 7)

        tf_h = self.enc_tf2(tf_h)
        re_h = self.enc_re2(re_h)

        comb = torch.cat([tf_h, re_h], dim=1)
        comb = self.combine(comb)

        pred = comb.permute(0, 2, 1)     # [B, d, TF+RE]
        pred = self.out1(pred)           # [B, d, TG]
        pred = pred.permute(0, 2, 1)     # [B, TG, d]
        pred = F.relu(pred)
        pred = self.out2(pred).squeeze(-1)  # [B, TG]

        align_loss = 0.5 * (a1 + a2)
        return pred, align_loss


class RETGModel(nn.Module):
    """
    Flow 2:
      Inputs: RE (accessibility) + TG (expression)
      Output: TF prediction

    Cross-attn:
      RE attends to TG (RE->TG)
      TG attends to RE (TG->RE)
    Alignment loss = 0.5 * (align_re2tg + align_tg2re)
    """
    def __init__(
        self,
        num_RE: int,
        num_TG: int,
        num_TF: int,
        cfg: Config,
        shared_re: nn.ModuleDict,
        re2tg_block: SparseCrossAttnFFNBlock,
        tg2re_block: SparseCrossAttnFFNBlock
    ):
        super().__init__()
        self.num_RE = num_RE
        self.num_TG = num_TG
        self.num_TF = num_TF
        self.cfg = cfg
        d = cfg.embedding_dim

        # Shared RE embeddings
        self.re_id = shared_re["id"]
        self.re_val = shared_re["val"]
        self.re_merge = shared_re["merge"]

        # TG embeddings
        self.tg_id = nn.Embedding(num_TG, d)
        self.tg_val = nn.Sequential(nn.Linear(1, d), nn.ReLU())
        self.tg_merge = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())

        # Self encoders
        self.enc_re1 = make_encoder(d, num_RE, cfg)
        self.enc_tg1 = make_encoder(d, num_TG, cfg)
        self.enc_re2 = make_encoder(d, num_RE, cfg)
        self.enc_tg2 = make_encoder(d, num_TG, cfg)

        # Sparse cross blocks
        self.cross_re = re2tg_block
        self.cross_tg = tg2re_block

        # Combine encoder
        self.combine = make_encoder(d, num_RE + num_TG, cfg)

        # Output head: map (RE+TG tokens) -> TF
        self.out1 = nn.Linear(num_RE + num_TG, num_TF)
        self.out2 = nn.Linear(d, 1)

    def forward(self, re_in: torch.Tensor, tg_in: torch.Tensor, seed: int):
        B = re_in.size(0)
        device = re_in.device

        re_idx = torch.arange(self.num_RE, device=device).unsqueeze(0).expand(B, -1)
        tg_idx = torch.arange(self.num_TG, device=device).unsqueeze(0).expand(B, -1)

        re_emb = self.re_merge(torch.cat([self.re_id(re_idx), self.re_val(re_in.unsqueeze(-1))], dim=-1))
        tg_emb = self.tg_merge(torch.cat([self.tg_id(tg_idx), self.tg_val(tg_in.unsqueeze(-1))], dim=-1))

        re_h = self.enc_re1(re_emb)
        tg_h = self.enc_tg1(tg_emb)

        re_h, a1 = self.cross_re(re_h, tg_h, compute_align=True, seed=seed)
        tg_h, a2 = self.cross_tg(tg_h, re_h, compute_align=True, seed=seed + 11)

        re_h = self.enc_re2(re_h)
        tg_h = self.enc_tg2(tg_h)

        comb = torch.cat([re_h, tg_h], dim=1)
        comb = self.combine(comb)

        pred = comb.permute(0, 2, 1)     # [B, d, RE+TG]
        pred = self.out1(pred)           # [B, d, TF]
        pred = pred.permute(0, 2, 1)     # [B, TF, d]
        pred = F.relu(pred)
        pred = self.out2(pred).squeeze(-1)  # [B, TF]

        align_loss = 0.5 * (a1 + a2)
        return pred, align_loss


class GRaSPJointSparse(nn.Module):
    """
    Joint wrapper that shares RE embedding parameters.
      Flow1: TF + RE -> TG
      Flow2: RE + TG -> TF
    """
    def __init__(
        self,
        num_TF: int,
        num_RE: int,
        num_TG: int,
        cfg: Config,
        tf2re_idx: torch.Tensor, tf2re_valid: torch.Tensor,
        re2tf_idx: torch.Tensor, re2tf_valid: torch.Tensor,
        re2tg_idx: torch.Tensor, re2tg_valid: torch.Tensor,
        tg2re_idx: torch.Tensor, tg2re_valid: torch.Tensor
    ):
        super().__init__()
        d = cfg.embedding_dim

        # Shared RE embedding components
        self.shared_re = nn.ModuleDict({
            "id": nn.Embedding(num_RE, d),
            "val": nn.Sequential(nn.Linear(1, d), nn.ReLU()),
            "merge": nn.Sequential(nn.Linear(2 * d, d), nn.ReLU()),
        })

        # Build sparse attention modules
        tf2re_attn = SparseNeighborAttention(
            d, cfg.num_heads, cfg.dropout, tf2re_idx, tf2re_valid,
            key_len=num_RE, neg_k=cfg.neg_k, neg_resample_steps=cfg.neg_resample_steps
        )
        re2tf_attn = SparseNeighborAttention(
            d, cfg.num_heads, cfg.dropout, re2tf_idx, re2tf_valid,
            key_len=num_TF, neg_k=cfg.neg_k, neg_resample_steps=cfg.neg_resample_steps
        )
        re2tg_attn = SparseNeighborAttention(
            d, cfg.num_heads, cfg.dropout, re2tg_idx, re2tg_valid,
            key_len=num_TG, neg_k=cfg.neg_k, neg_resample_steps=cfg.neg_resample_steps
        )
        tg2re_attn = SparseNeighborAttention(
            d, cfg.num_heads, cfg.dropout, tg2re_idx, tg2re_valid,
            key_len=num_RE, neg_k=cfg.neg_k, neg_resample_steps=cfg.neg_resample_steps
        )

        # Wrap attention with FFN blocks
        tf2re_block = SparseCrossAttnFFNBlock(tf2re_attn, cfg.dim_feedforward, cfg.dropout, cfg.activation)
        re2tf_block = SparseCrossAttnFFNBlock(re2tf_attn, cfg.dim_feedforward, cfg.dropout, cfg.activation)
        re2tg_block = SparseCrossAttnFFNBlock(re2tg_attn, cfg.dim_feedforward, cfg.dropout, cfg.activation)
        tg2re_block = SparseCrossAttnFFNBlock(tg2re_attn, cfg.dim_feedforward, cfg.dropout, cfg.activation)

        # Two flow models
        self.model1 = TFREModel(num_TF, num_RE, num_TG, cfg, self.shared_re, tf2re_block, re2tf_block)
        self.model2 = RETGModel(num_RE, num_TG, num_TF, cfg, self.shared_re, re2tg_block, tg2re_block)

    def forward(self, tf_in: torch.Tensor, re_in: torch.Tensor, tg_in: torch.Tensor, seed: int):
        pred_tg, align1 = self.model1(tf_in, re_in, seed=seed)
        pred_tf, align2 = self.model2(re_in, tg_in, seed=seed + 1000)
        return pred_tg, align1, pred_tf, align2


# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optimizer, scheduler, cfg: Config, device, global_step: int) -> Tuple[float, int]:
    """Train one epoch and return average loss."""
    model.train()
    mse = nn.MSELoss()
    total = 0.0
    steps = 0

    for tf_in, re_in, tg_in in loader:
        tf_in = tf_in.to(device)
        re_in = re_in.to(device)
        tg_in = tg_in.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred_tg, align1, pred_tf, align2 = model(tf_in, re_in, tg_in, seed=cfg.seed + global_step * 13)

        loss1 = mse(pred_tg, tg_in) + cfg.lambda_align * align1
        loss2 = mse(pred_tf, tf_in) + cfg.lambda_align * align2
        loss = cfg.alpha * loss1 + (1.0 - cfg.alpha) * loss2

        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total += loss.item()
        steps += 1
        global_step += 1

    return total / max(steps, 1), global_step


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> float:
    """
    Evaluate on test set.
    NOTE: We only report prediction MSE here by default, avoiding using "prior fit" as a generalization metric.
    """
    model.eval()
    mse = nn.MSELoss()
    total = 0.0
    steps = 0

    for tf_in, re_in, tg_in in loader:
        tf_in = tf_in.to(device)
        re_in = re_in.to(device)
        tg_in = tg_in.to(device)

        pred_tg, _, pred_tf, _ = model(tf_in, re_in, tg_in, seed=cfg.seed + 999)

        loss = cfg.alpha * mse(pred_tg, tg_in) + (1.0 - cfg.alpha) * mse(pred_tf, tf_in)
        total += loss.item()
        steps += 1

    return total / max(steps, 1)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing train/test CSVs and priors.")
    parser.add_argument("--out_dir", type=str, default="runs/grasp_sparse_joint", help="Output directory for checkpoints.")
    parser.add_argument("--save_best_only", action="store_true", help="If set, only save best_model.pt at the end.")

    # Training overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lambda_align", type=float, default=None)

    # Sparse controls
    parser.add_argument("--max_neighbors", type=int, default=None)
    parser.add_argument("--neg_k", type=int, default=None)

    # Logging
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)

    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.lambda_align is not None:
        cfg.lambda_align = args.lambda_align
    if args.max_neighbors is not None:
        cfg.max_neighbors = args.max_neighbors
    if args.neg_k is not None:
        cfg.neg_k = args.neg_k
    if args.use_wandb:
        cfg.use_wandb = True
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project

    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Linformer available: {LINFORMER_AVAILABLE}")

    # Load data
    (train_tf, train_re, train_tg), (test_tf, test_re, test_tg), (tf_names, re_names, tg_names) = load_data(args.data_dir)
    num_TF, num_RE, num_TG = len(tf_names), len(re_names), len(tg_names)
    print(f"Dims: TF={num_TF}, RE={num_RE}, TG={num_TG}")

    # Load priors (COO)
    print("Loading priors (COO edges)...")
    motif_rows_re, motif_cols_tf, _ = load_tf_re_motif_edges(args.data_dir, re_names, tf_names)  # RE->TF
    tss_rows_tg, tss_cols_re, _ = load_tg_re_tss_edges(args.data_dir, tg_names, re_names)       # TG->RE

    # Build neighbor tables (padded)
    print("Building neighbor tables (padded)...")
    # TF->RE from motif (swap)
    tf2re_idx, tf2re_valid = build_neighbor_table_from_coo(
        num_q=num_TF, num_k=num_RE,
        rows=motif_cols_tf, cols=motif_rows_re,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 1,
        fallback_empty="none",
    )
    # RE->TF from motif
    re2tf_idx, re2tf_valid = build_neighbor_table_from_coo(
        num_q=num_RE, num_k=num_TF,
        rows=motif_rows_re, cols=motif_cols_tf,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 2,
        fallback_empty="none",
    )
    # RE->TG from TSS (swap)
    re2tg_idx, re2tg_valid = build_neighbor_table_from_coo(
        num_q=num_RE, num_k=num_TG,
        rows=tss_cols_re, cols=tss_rows_tg,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 3,
        fallback_empty="none",
    )
    # TG->RE from TSS
    tg2re_idx, tg2re_valid = build_neighbor_table_from_coo(
        num_q=num_TG, num_k=num_RE,
        rows=tss_rows_tg, cols=tss_cols_re,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 4,
        fallback_empty="none",
    )

    # Scaling (fit train only) + SAVE scalers and feature names
    print("Scaling (fit on train only)...")
    (train_tf_s, train_re_s, train_tg_s), (test_tf_s, test_re_s, test_tg_s), scalers = fit_and_transform_scalers(
        train_tf, train_re, train_tg, test_tf, test_re, test_tg
    )

    # Save scalers (critical for interpret.py to reproduce scaling exactly)
    joblib.dump(scalers["tf"], os.path.join(args.out_dir, "scaler_tf.joblib"))
    joblib.dump(scalers["re"], os.path.join(args.out_dir, "scaler_re.joblib"))
    joblib.dump(scalers["tg"], os.path.join(args.out_dir, "scaler_tg.joblib"))

    # Save feature names
    with open(os.path.join(args.out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump({"tf": tf_names, "re": re_names, "tg": tg_names}, f, ensure_ascii=False, indent=2)

    # Save run config
    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    # DataLoaders
    train_ds = JointDataset(train_tf_s, train_re_s, train_tg_s)
    test_ds = JointDataset(test_tf_s, test_re_s, test_tg_s)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Model
    print("Initializing model...")
    model = GRaSPJointSparse(
        num_TF=num_TF, num_RE=num_RE, num_TG=num_TG, cfg=cfg,
        tf2re_idx=tf2re_idx, tf2re_valid=tf2re_valid,
        re2tf_idx=re2tf_idx, re2tf_valid=re2tf_valid,
        re2tg_idx=re2tg_idx, re2tg_valid=re2tg_valid,
        tg2re_idx=tg2re_idx, tg2re_valid=tg2re_valid,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup = min(cfg.warmup_steps, max(1, total_steps // 10))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    # wandb
    if cfg.use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not available. Proceed without wandb.")
            cfg.use_wandb = False
        else:
            wandb.init(project=cfg.wandb_project, config=cfg.__dict__)

    # Training loop
    print("Start training (training-only; IG moved to interpret.py)...")
    best_test = float("inf")
    global_step = 0

    for epoch in range(cfg.epochs):
        train_loss, global_step = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, device, global_step)
        test_loss = evaluate(model, test_loader, cfg, device)

        if cfg.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | test_loss={test_loss:.6f}")

        # Save best
        if test_loss < best_test:
            best_test = test_loss
            ckpt = {
                "config": cfg.__dict__,
                "state_dict": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
                "best_test_loss": best_test,
            }
            torch.save(ckpt, os.path.join(args.out_dir, "best_model.pt"))

    # Save final
    if not args.save_best_only:
        ckpt = {
            "config": cfg.__dict__,
            "state_dict": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
            "best_test_loss": best_test,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "final_model.pt"))

    if cfg.use_wandb:
        wandb.finish()

    print("Training finished.")
    print(f"Best test loss: {best_test:.6f}")
    print(f"Outputs saved to: {args.out_dir}")
    print("Run interpretability separately with interpret.py.")


if __name__ == "__main__":
    main()

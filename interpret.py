#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Captum
try:
    from captum.attr import IntegratedGradients
except Exception as e:
    raise ImportError("captum not installed or unavailable, please install it first: pip install captum") from e

import joblib

from main import (
    Config,
    load_expression_matrix,
    load_data,
    load_tf_re_motif_edges,
    load_tg_re_tss_edges,
    build_neighbor_table_from_coo,
    GRaSPJointSparse,
    seed_everything,
)

# -------------------------
# helpers
# -------------------------
def read_cell_types(path: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0)
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str)
    if "CellType" in df.columns:
        return df["CellType"].astype(str)
    # fallback: first column
    return df.iloc[:, 0].astype(str)

def filter_original_cells(sample_ids: pd.Index) -> np.ndarray:
    sid = sample_ids.astype(str)
    mask = ~sid.str.startswith("metacell_")
    mask &= ~sid.str.contains("_aug_")
    return mask.values

def parse_indices(arg: Optional[str], max_len: int) -> Optional[List[int]]:
    if arg is None:
        return None
    s = arg.strip()
    if ":" in s:
        a, b = s.split(":")
        a = int(a) if a else 0
        b = int(b) if b else max_len
        b = min(b, max_len)
        return list(range(a, b))
    parts = [p for p in s.split(",") if p.strip() != ""]
    out = [int(p) for p in parts]
    out = [i for i in out if 0 <= i < max_len]
    return out

def topk_edges(
    src_names: List[str],
    tgt_names: List[str],
    attr_matrix: np.ndarray,   # [Ntgt, Nsrc]
    topk: int,
    out_csv: str,
    prefix: str,
):
    rows = []
    Ntgt, Nsrc = attr_matrix.shape
    k = min(topk, Nsrc)
    for t in range(Ntgt):
        v = attr_matrix[t]
        idx = np.argpartition(np.abs(v), -k)[-k:]
        idx = idx[np.argsort(-np.abs(v[idx]))]
        for s in idx:
            rows.append({
                "edge_type": prefix,
                "source": src_names[s],
                "target": tgt_names[t],
                "attribution": float(v[s]),
                "abs_attribution": float(abs(v[s])),
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# -------------------------
# forward wrappers (no align sampling)
# -------------------------
@torch.no_grad()
def make_metacell_scaled(
    train_tf: pd.DataFrame,
    train_re: pd.DataFrame,
    train_tg: pd.DataFrame,
    cell_types: pd.Series,
    scaler_tf,
    scaler_re,
    scaler_tg,
    cell_type: str,
    use_original_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:

    common = train_tf.index.intersection(train_re.index).intersection(train_tg.index).intersection(cell_types.index)
    train_tf = train_tf.loc[common]
    train_re = train_re.loc[common]
    train_tg = train_tg.loc[common]
    cell_types = cell_types.loc[common]

    if use_original_only:
        mask_orig = filter_original_cells(common)
        train_tf = train_tf.iloc[mask_orig]
        train_re = train_re.iloc[mask_orig]
        train_tg = train_tg.iloc[mask_orig]
        cell_types = cell_types.iloc[mask_orig]
        common = train_tf.index

    sel = (cell_types.values == cell_type)
    if sel.sum() == 0:
        raise ValueError(f"cell_type='{cell_type}' not found in train_cell_types (or filtered out).")
    sel_ids = common[sel]

    tf_mean = train_tf.loc[sel_ids].mean(axis=0).values.reshape(1, -1)
    re_mean = train_re.loc[sel_ids].mean(axis=0).values.reshape(1, -1)
    tg_mean = train_tg.loc[sel_ids].mean(axis=0).values.reshape(1, -1)

    tf_s = scaler_tf.transform(tf_mean).astype(np.float32).reshape(-1)
    re_s = scaler_re.transform(re_mean).astype(np.float32).reshape(-1)
    tg_s = scaler_tg.transform(tg_mean).astype(np.float32).reshape(-1)

    info = [
        f"selected_cells={sel.sum()}",
        f"use_original_only={use_original_only}",
    ]
    return tf_s, re_s, tg_s, sel_ids.values, info

def predict_model1_noalign(joint, tf_in: torch.Tensor, re_in: torch.Tensor, seed: int) -> torch.Tensor:

    m = joint.model1
    B = tf_in.size(0)
    device = tf_in.device

    tf_idx = torch.arange(m.num_TF, device=device).unsqueeze(0).expand(B, -1)
    re_idx = torch.arange(m.num_RE, device=device).unsqueeze(0).expand(B, -1)

    tf_emb = m.tf_merge(torch.cat([m.tf_id(tf_idx), m.tf_val(tf_in.unsqueeze(-1))], dim=-1))
    re_emb = m.re_merge(torch.cat([m.re_id(re_idx), m.re_val(re_in.unsqueeze(-1))], dim=-1))

    tf_h = m.enc_tf1(tf_emb)
    re_h = m.enc_re1(re_emb)

    tf_h, _ = m.cross_tf(tf_h, re_h, compute_align=False, seed=seed)
    re_h, _ = m.cross_re(re_h, tf_h, compute_align=False, seed=seed + 7)

    tf_h = m.enc_tf2(tf_h)
    re_h = m.enc_re2(re_h)

    comb = torch.cat([tf_h, re_h], dim=1)
    comb = m.combine(comb)

    pred = comb.permute(0, 2, 1)
    pred = m.out1(pred)
    pred = pred.permute(0, 2, 1)
    pred = torch.relu(pred)
    pred = m.out2(pred).squeeze(-1)
    return pred

def predict_model2_noalign(joint, re_in: torch.Tensor, tg_in: torch.Tensor, seed: int) -> torch.Tensor:

    m = joint.model2
    B = re_in.size(0)
    device = re_in.device

    re_idx = torch.arange(m.num_RE, device=device).unsqueeze(0).expand(B, -1)
    tg_idx = torch.arange(m.num_TG, device=device).unsqueeze(0).expand(B, -1)

    re_emb = m.re_merge(torch.cat([m.re_id(re_idx), m.re_val(re_in.unsqueeze(-1))], dim=-1))
    tg_emb = m.tg_merge(torch.cat([m.tg_id(tg_idx), m.tg_val(tg_in.unsqueeze(-1))], dim=-1))

    re_h = m.enc_re1(re_emb)
    tg_h = m.enc_tg1(tg_emb)

    re_h, _ = m.cross_re(re_h, tg_h, compute_align=False, seed=seed)
    tg_h, _ = m.cross_tg(tg_h, re_h, compute_align=False, seed=seed + 11)

    re_h = m.enc_re2(re_h)
    tg_h = m.enc_tg2(tg_h)

    comb = torch.cat([re_h, tg_h], dim=1)
    comb = m.combine(comb)

    pred = comb.permute(0, 2, 1)
    pred = m.out1(pred)
    pred = pred.permute(0, 2, 1)
    pred = torch.relu(pred)
    pred = m.out2(pred).squeeze(-1)
    return pred

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="aug_out", help="data_augmentation.py output directory")
    ap.add_argument("--run_dir", type=str, required=True, help="training output directory (contains best_model.pt / final_model.pt / scaler_*.joblib)")
    ap.add_argument("--ckpt", type=str, default="best_model.pt", help="best_model.pt or final_model.pt")
    ap.add_argument("--cell_type", type=str, required=True, help="specified cell type name (consistent with train_cell_types.csv)")
    ap.add_argument("--use_original_only", action="store_true", help="use only original cells (default True)")
    ap.add_argument("--include_augmented", action="store_true", help="include augmented samples (overrides use_original_only)")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--n_steps", type=int, default=50, help="IG steps")
    ap.add_argument("--internal_batch_size", type=int, default=1, help="IG internal batch size (usually 1 to save memory)")

    ap.add_argument("--tg_targets", type=str, default=None, help="TG indices to interpret, e.g., '0:100' or '0,1,2'")
    ap.add_argument("--tf_targets", type=str, default=None, help="TF indices to interpret, e.g., '0:100' or '0,1,2'")

    ap.add_argument("--topk", type=int, default=200, help="topK attribution edges to output per target")
    ap.add_argument("--save_npz", action="store_true", help="also save dense attribution matrix as npz (can be very large for big data)")
    args = ap.parse_args()

    run_dir = args.run_dir
    ckpt_path = os.path.join(run_dir, args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # load scalers
    scaler_tf = joblib.load(os.path.join(run_dir, "scaler_tf.joblib"))
    scaler_re = joblib.load(os.path.join(run_dir, "scaler_re.joblib"))
    scaler_tg = joblib.load(os.path.join(run_dir, "scaler_tg.joblib"))

    # load features (optional)
    feat_json = os.path.join(run_dir, "feature_names.json")
    feature_names = None
    if os.path.exists(feat_json):
        with open(feat_json, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

    # load data
    (train_tf, train_re, train_tg), (_, _, _), (tf_names, re_names, tg_names) = load_data(args.data_dir)
    if feature_names is not None:
        # The names saved during training should match the data; if not, use the data names
        pass

    # cell types
    ct_path = os.path.join(args.data_dir, "train_cell_types.csv")
    if not os.path.exists(ct_path):
        raise FileNotFoundError(f"找不到 train_cell_types.csv: {ct_path}")
    cell_types = read_cell_types(ct_path)

    use_original_only = True
    if args.include_augmented:
        use_original_only = False
    if args.use_original_only:
        use_original_only = True

    # metacell scaled
    tf_s, re_s, tg_s, sel_ids, info = make_metacell_scaled(
        train_tf, train_re, train_tg, cell_types,
        scaler_tf, scaler_re, scaler_tg,
        cell_type=args.cell_type,
        use_original_only=use_original_only,
    )
    print("[Metacell]", args.cell_type, "|", "; ".join(info))

    # device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("Device:", device)

    # load ckpt config/state
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    cfg = Config(**cfg_dict) if cfg_dict else Config()
    seed_everything(cfg.seed)

    # Build priors + neighbor tables (must be consistent with training to correctly instantiate model and load state_dict)
    # Note: using cfg.max_neighbors (training-time pruning) here to ensure buffer shape consistency
    motif_rows_re, motif_cols_tf, _ = load_tf_re_motif_edges(args.data_dir, re_names, tf_names)  # RE->TF
    tss_rows_tg, tss_cols_re, _ = load_tg_re_tss_edges(args.data_dir, tg_names, re_names)       # TG->RE

    tf2re_idx, tf2re_valid = build_neighbor_table_from_coo(
        num_q=len(tf_names), num_k=len(re_names),
        rows=motif_cols_tf, cols=motif_rows_re,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 1,
        fallback_empty="none"
    )
    re2tf_idx, re2tf_valid = build_neighbor_table_from_coo(
        num_q=len(re_names), num_k=len(tf_names),
        rows=motif_rows_re, cols=motif_cols_tf,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 2,
        fallback_empty="none"
    )
    re2tg_idx, re2tg_valid = build_neighbor_table_from_coo(
        num_q=len(re_names), num_k=len(tg_names),
        rows=tss_cols_re, cols=tss_rows_tg,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 3,
        fallback_empty="none"
    )
    tg2re_idx, tg2re_valid = build_neighbor_table_from_coo(
        num_q=len(tg_names), num_k=len(re_names),
        rows=tss_rows_tg, cols=tss_cols_re,
        max_neighbors=cfg.max_neighbors, seed=cfg.seed + 4,
        fallback_empty="none"
    )

    # instantiate model + load weights
    model = GRaSPJointSparse(
        num_TF=len(tf_names), num_RE=len(re_names), num_TG=len(tg_names), cfg=cfg,
        tf2re_idx=tf2re_idx, tf2re_valid=tf2re_valid,
        re2tf_idx=re2tf_idx, re2tf_valid=re2tf_valid,
        re2tg_idx=re2tg_idx, re2tg_valid=re2tg_valid,
        tg2re_idx=tg2re_idx, tg2re_valid=tg2re_valid
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)

    # prepare tensors (scaled), baseline zeros (因为 StandardScaler: 0 对应训练均值)
    tf_x = torch.tensor(tf_s, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    re_x = torch.tensor(re_s, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    tg_x = torch.tensor(tg_s, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)

    tf_base = torch.zeros_like(tf_x)
    re_base = torch.zeros_like(re_x)
    tg_base = torch.zeros_like(tg_x)

    # targets
    tg_targets = parse_indices(args.tg_targets, max_len=len(tg_names))
    tf_targets = parse_indices(args.tf_targets, max_len=len(tf_names))
    if tg_targets is None:
        tg_targets = list(range(len(tg_names)))
    if tf_targets is None:
        tf_targets = list(range(len(tf_names)))

    # output dir
    out_dir = os.path.join(run_dir, "interpret", args.cell_type.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # IG: Model1 (TF,RE -> TG)
    # -------------------------
    print("\n[IG] Model1: TF/RE -> TG")

    def f_model1_tf(tf_in):
        return predict_model1_noalign(model, tf_in, re_x.detach(), seed=cfg.seed + 777)

    def f_model1_re(re_in):
        return predict_model1_noalign(model, tf_x.detach(), re_in, seed=cfg.seed + 777)

    ig_tf = IntegratedGradients(f_model1_tf)
    ig_re = IntegratedGradients(f_model1_re)

    tf_attr = []
    re_attr = []

    for i, tg_idx in enumerate(tg_targets):
        a_tf = ig_tf.attribute(
            inputs=tf_x,
            baselines=tf_base,
            target=tg_idx,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            return_convergence_delta=False,
        )
        a_re = ig_re.attribute(
            inputs=re_x,
            baselines=re_base,
            target=tg_idx,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            return_convergence_delta=False,
        )
        tf_attr.append(a_tf.squeeze(0).detach().cpu().numpy())
        re_attr.append(a_re.squeeze(0).detach().cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"  processed TG targets: {i+1}/{len(tg_targets)}")

    tf_attr = np.stack(tf_attr, axis=0)  # [Ntgt, TF]
    re_attr = np.stack(re_attr, axis=0)  # [Ntgt, RE]
    tg_names_sel = [tg_names[i] for i in tg_targets]

    # save topk edge lists (recommended)
    topk_edges(tf_names, tg_names_sel, tf_attr, args.topk,
               os.path.join(out_dir, "IG_TF_to_TG_topk.csv"), prefix="TF->TG")
    topk_edges(re_names, tg_names_sel, re_attr, args.topk,
               os.path.join(out_dir, "IG_RE_to_TG_topk.csv"), prefix="RE->TG")

    if args.save_npz:
        np.savez_compressed(os.path.join(out_dir, "IG_TF_to_TG.npz"),
                            tf_attr=tf_attr, tg_targets=np.array(tg_targets, dtype=np.int32))
        np.savez_compressed(os.path.join(out_dir, "IG_RE_to_TG.npz"),
                            re_attr=re_attr, tg_targets=np.array(tg_targets, dtype=np.int32))

    # -------------------------
    # IG: Model2 (RE,TG -> TF)
    # -------------------------
    print("\n[IG] Model2: RE/TG -> TF")

    def f_model2_re(re_in):
        return predict_model2_noalign(model, re_in, tg_x.detach(), seed=cfg.seed + 888)

    def f_model2_tg(tg_in):
        return predict_model2_noalign(model, re_x.detach(), tg_in, seed=cfg.seed + 888)

    ig_re2 = IntegratedGradients(f_model2_re)
    ig_tg2 = IntegratedGradients(f_model2_tg)

    re_attr2 = []
    tg_attr2 = []

    for i, tf_idx in enumerate(tf_targets):
        a_re2 = ig_re2.attribute(
            inputs=re_x,
            baselines=re_base,
            target=tf_idx,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            return_convergence_delta=False,
        )
        a_tg2 = ig_tg2.attribute(
            inputs=tg_x,
            baselines=tg_base,
            target=tf_idx,
            n_steps=args.n_steps,
            internal_batch_size=args.internal_batch_size,
            return_convergence_delta=False,
        )
        re_attr2.append(a_re2.squeeze(0).detach().cpu().numpy())
        tg_attr2.append(a_tg2.squeeze(0).detach().cpu().numpy())

        if (i + 1) % 50 == 0:
            print(f"  processed TF targets: {i+1}/{len(tf_targets)}")

    re_attr2 = np.stack(re_attr2, axis=0)  # [Ntf, RE]
    tg_attr2 = np.stack(tg_attr2, axis=0)  # [Ntf, TG]
    tf_names_sel = [tf_names[i] for i in tf_targets]

    topk_edges(re_names, tf_names_sel, re_attr2, args.topk,
               os.path.join(out_dir, "IG_RE_to_TF_topk.csv"), prefix="RE->TF")
    topk_edges(tg_names, tf_names_sel, tg_attr2, args.topk,
               os.path.join(out_dir, "IG_TG_to_TF_topk.csv"), prefix="TG->TF")

    if args.save_npz:
        np.savez_compressed(os.path.join(out_dir, "IG_RE_to_TF.npz"),
                            re_attr=re_attr2, tf_targets=np.array(tf_targets, dtype=np.int32))
        np.savez_compressed(os.path.join(out_dir, "IG_TG_to_TF.npz"),
                            tg_attr=tg_attr2, tf_targets=np.array(tf_targets, dtype=np.int32))

    # save metacell info
    with open(os.path.join(out_dir, "metacell_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "cell_type": args.cell_type,
            "selected_cell_ids_head": sel_ids[:20].tolist(),
            "selected_cell_count": int(len(sel_ids)),
            "use_original_only": bool(use_original_only),
            "tg_targets": tg_targets[:50],
            "tf_targets": tf_targets[:50],
        }, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print("Outputs saved to:", out_dir)


if __name__ == "__main__":
    main()

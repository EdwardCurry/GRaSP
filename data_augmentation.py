import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _read_celltype(cell_type_df: pd.DataFrame) -> pd.Series:
    """
    Normalize cell type dataframe -> Series[name='CellType'] with index = cell_id
    """
    if isinstance(cell_type_df, pd.Series):
        s = cell_type_df.copy()
        s.name = "CellType"
        return s

    # common column names fallback
    if "CellType" in cell_type_df.columns:
        s = cell_type_df["CellType"].copy()
        s.name = "CellType"
        return s

    if cell_type_df.shape[1] == 1:
        s = cell_type_df.iloc[:, 0].copy()
        s.name = "CellType"
        return s

    raise ValueError("cell_type_final.csv must contain a 'CellType' column (or only one column).")

def _align_modalities(scRNA: pd.DataFrame, scATAC: pd.DataFrame, cell_types: pd.Series):
    """
    Strictly align by common cell IDs, keeping deterministic order.
    """
    # keep order of scRNA columns
    common = scRNA.columns.intersection(scATAC.columns)
    common = common.intersection(cell_types.index)

    # pandas Index.intersection may sort depending on version; enforce scRNA order:
    common = [c for c in scRNA.columns if (c in set(common))]

    scRNA = scRNA.loc[:, common]
    scATAC = scATAC.loc[:, common]
    cell_types = cell_types.loc[common]

    return scRNA, scATAC, cell_types

def _handle_rare_classes_for_stratify(cell_types: pd.Series, min_count: int = 2):
    """
    train_test_split with stratify requires each class count >= 2 (and also depends on test_size).
    We do a safe fallback:
      - If any class count < min_count: merge them into 'Other'
    """
    counts = cell_types.value_counts()
    rare = counts[counts < min_count].index.tolist()
    if len(rare) == 0:
        return cell_types, False

    merged = cell_types.copy()
    merged.loc[merged.isin(rare)] = "Other"
    return merged, True

# -----------------------------
# Load + Split
# -----------------------------
def load_and_align_data(
    rna_path="scRNA_data_final.csv",
    atac_path="scATAC_data_final.csv",
    celltype_path="cell_type_final.csv",
):
    print("Loading raw data...")
    scRNA = pd.read_csv(rna_path, index_col=0)
    scATAC = pd.read_csv(atac_path, index_col=0)
    ct_df = pd.read_csv(celltype_path, index_col=0)

    cell_types = _read_celltype(ct_df)

    scRNA, scATAC, cell_types = _align_modalities(scRNA, scATAC, cell_types)
    print(f"Aligned cells: {scRNA.shape[1]} | Genes: {scRNA.shape[0]} | Peaks: {scATAC.shape[0]}")
    return scRNA, scATAC, cell_types

def split_data(scRNA: pd.DataFrame, scATAC: pd.DataFrame, cell_types: pd.Series, test_size=0.2):
    """
    CRITICAL: split BEFORE augmentation.
    """
    print(f"Splitting data: Train {int((1-test_size)*100)}% / Test {int(test_size*100)}% (before augmentation)")
    cells = scRNA.columns.to_list()

    strat_y, merged = _handle_rare_classes_for_stratify(cell_types, min_count=2)
    if merged:
        print("Warning: rare cell types detected (<2). They are merged into 'Other' for stratified split.")

    # train_test_split expects y aligned with cells in same order
    y = strat_y.values

    try:
        train_cells, test_cells = train_test_split(
            cells,
            test_size=test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}). Fallback to non-stratified split.")
        train_cells, test_cells = train_test_split(
            cells,
            test_size=test_size,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    train_rna = scRNA[train_cells]
    train_atac = scATAC[train_cells]
    train_ct = cell_types.loc[train_cells]

    test_rna = scRNA[test_cells]
    test_atac = scATAC[test_cells]
    test_ct = cell_types.loc[test_cells]

    # Hard anti-leak check
    inter = set(train_cells).intersection(set(test_cells))
    assert len(inter) == 0, f"Leak detected: train/test overlap = {len(inter)}"

    print(f"Train cells: {len(train_cells)} | Test cells: {len(test_cells)}")
    return (train_rna, train_atac, train_ct), (test_rna, test_atac, test_ct)

# -----------------------------
# Augment (TRAIN ONLY)
# -----------------------------
def generate_metacells(
    scRNA: pd.DataFrame,
    scATAC: pd.DataFrame,
    cell_types: pd.Series,
    cells_per_metacell=5,
    times_per_cell=4,
    rng=None,
):
    """
    Metacells ONLY from training set.
    Returns:
      meta_rna_df, meta_atac_df, meta_ct_series, meta_comp_df
    meta_comp_df: metacell_id -> members (comma-separated)
    """
    print("Generating metacells (TRAIN ONLY)...")
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    meta_rna_cols = []
    meta_atac_cols = []
    meta_ct = []
    meta_ids = []
    meta_members = []

    metacell_index = 0
    for c_type in cell_types.unique():
        type_cells = cell_types.index[cell_types == c_type].to_list()
        n = len(type_cells)
        if n < cells_per_metacell:
            print(f"  - Skip {c_type}: n={n} < cells_per_metacell={cells_per_metacell}")
            continue

        requested = int(math.ceil((n * times_per_cell) / cells_per_metacell))
        max_combos = math.comb(n, cells_per_metacell)
        target = min(requested, max_combos)
        if target < requested:
            print(f"  - {c_type}: requested {requested} > max_combos {max_combos}, clipped to {target}")

        seen = set()
        attempts = 0
        max_attempts = max(100, target * 20)

        generated = 0
        while generated < target and attempts < max_attempts:
            attempts += 1
            group = tuple(sorted(rng.choice(type_cells, size=cells_per_metacell, replace=False)))
            if group in seen:
                continue
            seen.add(group)

            # mean across selected cells
            rna_meta = scRNA.loc[:, list(group)].mean(axis=1)
            atac_meta = scATAC.loc[:, list(group)].mean(axis=1)

            mc_id = f"MC__{c_type}__{metacell_index}"
            metacell_index += 1

            meta_rna_cols.append(rna_meta)
            meta_atac_cols.append(atac_meta)
            meta_ct.append(c_type)
            meta_ids.append(mc_id)
            meta_members.append(",".join(group))

            generated += 1

        print(f"  - {c_type}: generated {generated}/{target} metacells (attempts={attempts})")

    if len(meta_ids) == 0:
        print("No metacells generated.")
        return None, None, None, None

    meta_rna_df = pd.DataFrame(meta_rna_cols, index=meta_ids).T
    meta_atac_df = pd.DataFrame(meta_atac_cols, index=meta_ids).T
    meta_ct_series = pd.Series(meta_ct, index=meta_ids, name="CellType")

    meta_comp_df = pd.DataFrame({"members": meta_members}, index=meta_ids)

    # Ensure indices match
    meta_rna_df.index = scRNA.index
    meta_atac_df.index = scATAC.index

    return meta_rna_df, meta_atac_df, meta_ct_series, meta_comp_df

def shuffle_matching_data(
    scRNA: pd.DataFrame,
    scATAC: pd.DataFrame,
    cell_types: pd.Series,
    n_times=2,
    rng=None,
):
    """
    Matching shuffle ONLY within training set.
    For each shuffle round t, create new samples:
      RNA from cell i, ATAC from a permuted cell j (same cell type).
    New column names: {cell_id}__shuf{t}
    Also returns a mapping dataframe: new_id -> atac_source_cell
    """
    print("Applying matching shuffle (TRAIN ONLY)...")
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    rna_blocks = [scRNA]
    atac_blocks = [scATAC]
    ct_blocks = [cell_types]

    map_rows = []

    for t in range(1, n_times + 1):
        new_rna_parts = []
        new_atac_parts = []
        new_ct_parts = []

        for c_type in cell_types.unique():
            cells = cell_types.index[cell_types == c_type].to_list()
            if len(cells) == 0:
                continue
            perm = rng.permutation(cells)

            new_cols = [f"{c}__shuf{t}" for c in cells]

            # RNA stays with the destination cell
            rna_part = scRNA.loc[:, cells].copy()
            rna_part.columns = new_cols

            # ATAC comes from permuted source cells (but columns named as destination new_cols)
            atac_part = scATAC.loc[:, perm].copy()
            atac_part.columns = new_cols

            ct_part = pd.Series([c_type] * len(cells), index=new_cols, name="CellType")

            new_rna_parts.append(rna_part)
            new_atac_parts.append(atac_part)
            new_ct_parts.append(ct_part)

            # record mapping: destination -> source
            for dest_cell, src_cell, new_id in zip(cells, perm, new_cols):
                map_rows.append({"new_id": new_id, "dest_cell": dest_cell, "atac_source_cell": src_cell, "CellType": c_type})

        rna_blocks.append(pd.concat(new_rna_parts, axis=1))
        atac_blocks.append(pd.concat(new_atac_parts, axis=1))
        ct_blocks.append(pd.concat(new_ct_parts))

    final_rna = pd.concat(rna_blocks, axis=1)
    final_atac = pd.concat(atac_blocks, axis=1)
    final_ct = pd.concat(ct_blocks)

    # hard alignment check
    assert list(final_rna.columns) == list(final_atac.columns), "RNA/ATAC columns not aligned!"
    assert final_ct.index.isin(final_rna.columns).all(), "CellType index mismatch!"

    map_df = pd.DataFrame(map_rows)
    return final_rna, final_atac, final_ct, map_df

def separate_tf_tg(rna_data: pd.DataFrame, tf_list_file="TFName_human.txt"):
    print("Separating TF and TG expression...")
    with open(tf_list_file, "r") as f:
        tf_set = set([line.strip() for line in f if line.strip()])

    genes = rna_data.index.to_list()
    tf_genes = [g for g in genes if g in tf_set]
    tg_genes = [g for g in genes if g not in tf_set]

    tf_expr = rna_data.loc[tf_genes]
    tg_expr = rna_data.loc[tg_genes]
    return tf_expr, tg_expr

# -----------------------------
# Main
# -----------------------------
def main(out_dir="aug_out"):
    _ensure_dir(out_dir)
    rng = np.random.default_rng(RANDOM_STATE)

    # 1) Load + Align
    scRNA, scATAC, cell_types = load_and_align_data()

    # 2) Split BEFORE augmentation
    (train_rna, train_atac, train_ct), (test_rna, test_atac, test_ct) = split_data(scRNA, scATAC, cell_types)

    # Save split lists for strict reproducibility
    pd.Series(train_rna.columns, name="train_cells").to_csv(os.path.join(out_dir, "train_cells.csv"), index=False)
    pd.Series(test_rna.columns, name="test_cells").to_csv(os.path.join(out_dir, "test_cells.csv"), index=False)

    # 3) TRAIN augmentation
    meta_rna, meta_atac, meta_ct, meta_comp = generate_metacells(
        train_rna, train_atac, train_ct,
        cells_per_metacell=5,
        times_per_cell=4,
        rng=rng
    )

    if meta_rna is not None:
        train_rna2 = pd.concat([train_rna, meta_rna], axis=1)
        train_atac2 = pd.concat([train_atac, meta_atac], axis=1)
        train_ct2 = pd.concat([train_ct, meta_ct], axis=0)
        meta_comp.to_csv(os.path.join(out_dir, "metacell_composition.csv"))
    else:
        train_rna2, train_atac2, train_ct2 = train_rna, train_atac, train_ct

    train_rna_final, train_atac_final, train_ct_final, shuffle_map = shuffle_matching_data(
        train_rna2, train_atac2, train_ct2,
        n_times=2,
        rng=rng
    )
    shuffle_map.to_csv(os.path.join(out_dir, "shuffle_mapping.csv"), index=False)

    # 4) Anti-leak assertions
    inter = set(train_rna_final.columns).intersection(set(test_rna.columns))
    assert len(inter) == 0, f"Leak detected after augmentation: {len(inter)} overlapped columns!"

    # 5) TF/TG split (Train uses AUG, Test uses RAW)
    train_tf, train_tg = separate_tf_tg(train_rna_final)
    test_tf, test_tg = separate_tf_tg(test_rna)

    # 6) Save
    print("Saving datasets...")
    train_tf.to_csv(os.path.join(out_dir, "train_tf_expression.csv"))
    train_tg.to_csv(os.path.join(out_dir, "train_tg_expression.csv"))
    train_atac_final.to_csv(os.path.join(out_dir, "train_atac_data.csv"))
    train_ct_final.to_csv(os.path.join(out_dir, "train_cell_types.csv"), header=True)

    test_tf.to_csv(os.path.join(out_dir, "test_tf_expression.csv"))
    test_tg.to_csv(os.path.join(out_dir, "test_tg_expression.csv"))
    test_atac.to_csv(os.path.join(out_dir, "test_atac_data.csv"))
    test_ct.to_csv(os.path.join(out_dir, "test_cell_types.csv"), header=True)

    print("Done.")
    print(f"Final TRAIN samples (aug): {train_rna_final.shape[1]}")
    print(f"Final TEST  samples (raw): {test_rna.shape[1]}")

if __name__ == "__main__":
    main()

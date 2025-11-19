import pandas as pd
import numpy as np

with open('GeneName.txt', 'r') as file:
    gene_names = [line.strip() for line in file]

scRNA_data = pd.read_csv('PBMC_scRNA.txt', sep='\t', header=None)

scRNA_data.index = gene_names

with open('PeakName.txt', 'r') as file:
    peak_names = [line.strip() for line in file]

scATAC_data = pd.read_csv('PBMC_scATAC.txt', sep='\t', header=None)

scATAC_data.index = peak_names

with open('CellType.txt', 'r') as file:
    cell_types = [line.strip() for line in file]

num_cells_scRNA = scRNA_data.shape[1]
num_cells_scATAC = scATAC_data.shape[1]
num_cell_types = len(cell_types)

cell_ids = [f'Cell_{i}' for i in range(num_cells_scRNA)]
num_cell_ids=len(cell_ids)

scRNA_data.columns = cell_ids
scATAC_data.columns = cell_ids

gene_expression_counts = (scRNA_data > 0).sum(axis=1)

threshold = num_cells_scRNA * 0.01

genes_to_keep = gene_expression_counts[gene_expression_counts >= threshold].index
scRNA_data_filtered = scRNA_data.loc[genes_to_keep]

total_counts = scRNA_data_filtered.sum(axis=0)

scRNA_data_norm = scRNA_data_filtered.div(total_counts, axis=1) * 1e4

scRNA_data_log = np.log1p(scRNA_data_norm)

peak_accessibility_counts = (scATAC_data > 0).sum(axis=1)

threshold = num_cells_scATAC * 0.01

peaks_to_keep = peak_accessibility_counts[peak_accessibility_counts >= threshold].index
scATAC_data_filtered = scATAC_data.loc[peaks_to_keep]

tf = scATAC_data_filtered.div(scATAC_data_filtered.sum(axis=0), axis=1)

idf = np.log(1 + num_cells_scATAC / (scATAC_data_filtered > 0).sum(axis=1))

scATAC_data_tfidf = tf.mul(idf, axis=0)

cell_counts_scRNA = scRNA_data_filtered.sum(axis=0)

cell_threshold_scRNA = np.percentile(cell_counts_scRNA, 10)

cells_to_keep_scRNA = cell_counts_scRNA[cell_counts_scRNA >= cell_threshold_scRNA].index
scRNA_data_final = scRNA_data_log[cells_to_keep_scRNA]

cell_counts_scATAC = scATAC_data_filtered.sum(axis=0)

cell_threshold_scATAC = np.percentile(cell_counts_scATAC, 10)

cells_to_keep_scATAC = cell_counts_scATAC[cell_counts_scATAC >= cell_threshold_scATAC].index
scATAC_data_final = scATAC_data_tfidf[cells_to_keep_scATAC]

common_cells = scRNA_data_final.columns.intersection(scATAC_data_final.columns)

scRNA_data_final = scRNA_data_final[common_cells]
scATAC_data_final = scATAC_data_final[common_cells]

cell_type_df = pd.DataFrame({'CellID': cell_ids, 'CellType': cell_types})

cell_type_df.set_index('CellID', inplace=True)

cell_type_final = cell_type_df.loc[common_cells]

scRNA_data_final.to_csv('scRNA_data_final.csv')

scATAC_data_final.to_csv('scATAC_data_final.csv')

cell_type_final.to_csv('cell_type_final.csv')
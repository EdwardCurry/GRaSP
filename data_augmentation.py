import pandas as pd
import numpy as np
import scanpy as sc

scRNA_data_final = pd.read_csv('scRNA_data_final.csv', index_col=0)
scATAC_data_final = pd.read_csv('scATAC_data_final.csv', index_col=0)
cell_type_final = pd.read_csv('cell_type_final.csv', index_col=0)

random_state = 42

common_cells = scRNA_data_final.columns.intersection(scATAC_data_final.columns)
scRNA_data_final = scRNA_data_final[common_cells]
scATAC_data_final = scATAC_data_final[common_cells]
cell_type_final = cell_type_final.loc[common_cells]

def generate_metacells_multiple_usage(scRNA_data, scATAC_data, cell_types, cells_per_metacell=5, times_per_cell=4, random_state=None):
    rng = np.random.default_rng(random_state)
    metacell_scRNA_list = []
    metacell_scATAC_list = []
    metacell_cell_types = []
    metacell_ids = []
    metacell_index = 0

    unique_cell_types = cell_types.unique()
    for cell_type in unique_cell_types:
        cell_type_cells = cell_types[cell_types == cell_type].index
        num_cells = len(cell_type_cells)

        num_metacells = int(np.ceil((num_cells * times_per_cell) / cells_per_metacell))

        generated_combinations = set()

        for _ in range(num_metacells):
            while True:
                group_cells = tuple(rng.choice(cell_type_cells, size=cells_per_metacell, replace=False))
                if group_cells not in generated_combinations:
                    generated_combinations.add(group_cells)
                    break

            scRNA_metacell = scRNA_data[list(group_cells)].mean(axis=1)
            metacell_scRNA_list.append(scRNA_metacell)

            scATAC_metacell = scATAC_data[list(group_cells)].mean(axis=1)
            metacell_scATAC_list.append(scATAC_metacell)

            metacell_cell_types.append(cell_type)
            metacell_id = f'metacell_{metacell_index}'
            metacell_ids.append(metacell_id)
            metacell_index += 1

            if len(generated_combinations) >= num_cells * times_per_cell / cells_per_metacell:
                break

    metacell_scRNA_data = pd.DataFrame(metacell_scRNA_list).T
    metacell_scRNA_data.columns = metacell_ids
    metacell_scATAC_data = pd.DataFrame(metacell_scATAC_list).T
    metacell_scATAC_data.columns = metacell_ids
    metacell_cell_types = pd.Series(metacell_cell_types, index=metacell_ids)

    return metacell_scRNA_data, metacell_scATAC_data, metacell_cell_types

random_state = 42
metacell_scRNA_data, metacell_scATAC_data, metacell_cell_types = generate_metacells_multiple_usage(
    scRNA_data=scRNA_data_final,
    scATAC_data=scATAC_data_final,
    cell_types=cell_type_final['CellType'],
    cells_per_metacell=5,
    times_per_cell=4,  
    random_state=random_state
)


combined_scRNA_data = pd.concat([scRNA_data_final, metacell_scRNA_data], axis=1)
combined_scATAC_data = pd.concat([scATAC_data_final, metacell_scATAC_data], axis=1)
combined_cell_types = pd.concat([cell_type_final['CellType'], metacell_cell_types])

def shuffle_matching_data(scRNA_data, scATAC_data, cell_types, n_times=2, random_state=None):
    rng = np.random.default_rng(random_state)
    augmented_scRNA_list = [scRNA_data]
    augmented_scATAC_list = [scATAC_data]
    augmented_cell_types = [cell_types]

    for _ in range(n_times):

        shuffled_scATAC_data = scATAC_data.copy()
        for cell_type in cell_types.unique():

            cell_type_cells = cell_types[cell_types == cell_type].index

            shuffled_cells = rng.permutation(cell_type_cells)
            shuffled_scATAC_data[cell_type_cells] = scATAC_data[shuffled_cells].values

        augmented_scRNA_list.append(scRNA_data)
        augmented_scATAC_list.append(shuffled_scATAC_data)
        augmented_cell_types.append(cell_types)


    augmented_scRNA_data = pd.concat(augmented_scRNA_list, axis=1)
    augmented_scATAC_data = pd.concat(augmented_scATAC_list, axis=1)
    augmented_cell_types = pd.concat(augmented_cell_types)

    return augmented_scRNA_data, augmented_scATAC_data, augmented_cell_types

random_state = 42
augmented_scRNA_data, augmented_scATAC_data, augmented_cell_types = shuffle_matching_data(
    scRNA_data=combined_scRNA_data,
    scATAC_data=combined_scATAC_data,
    cell_types=combined_cell_types,
    n_times=2,
    random_state=random_state
)


augmented_scRNA_data.to_csv('augmented_scRNA_data.csv')


augmented_scATAC_data.to_csv('augmented_scATAC_data.csv')


augmented_cell_types.to_csv('augmented_cell_types.csv', header=True)


with open('TFName_human.txt', 'r') as f:
    tf_list = [line.strip() for line in f]


genes = augmented_scRNA_data.index.tolist()

gene_set = set(genes)


tf_in_data = [tf for tf in tf_list if tf in gene_set]


tg_in_data = [gene for gene in genes if gene not in tf_in_data]


tf_expression = augmented_scRNA_data.loc[tf_in_data]

tg_expression = augmented_scRNA_data.loc[tg_in_data]

tf_expression.to_csv('tf_expression.csv')

tg_expression.to_csv('tg_expression.csv')
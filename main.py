import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
import math
import os
import wandb
import warnings
from linformer import Linformer
from collections import Counter
from captum.attr import IntegratedGradients
from scipy.sparse import coo_matrix , csr_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)

tf_expression_file = 'tf_expression.csv'
re_expression_file = 'augmented_scATAC_data.csv'
tf_re_motif_file   = 'tf_re_motif.csv'
tg_expression_file = 'filtered_tg_expression.csv'
cell_types_file    = 'augmented_cell_types.csv' 
re_tg_matrix_file = 'tg_re_sparse_matrix.npz'  

tf_expression = pd.read_csv(tf_expression_file, index_col=0).T
tf_names = tf_expression.columns.tolist()
num_TF = len(tf_names)

re_expression = pd.read_csv(re_expression_file, index_col=0).T
re_names = re_expression.columns.tolist()
num_RE = len(re_names)


tg_expression = pd.read_csv(tg_expression_file, index_col=0).T
tg_names = tg_expression.columns.tolist()
num_TG = len(tg_names)

tf_re_motif_df = pd.read_csv(tf_re_motif_file, index_col=0)

all_zero_tfs = tf_re_motif_df.columns[(tf_re_motif_df == 0).all(axis=0)].tolist()
if all_zero_tfs:
    tf_re_motif_df = tf_re_motif_df.drop(columns=all_zero_tfs)
    tf_expression = tf_expression.drop(columns=all_zero_tfs)

all_zero_res = tf_re_motif_df.index[(tf_re_motif_df == 0).all(axis=1)].tolist()
if all_zero_res:
    tf_re_motif_df = tf_re_motif_df.drop(index=all_zero_res)
    re_expression = re_expression.drop(columns=all_zero_res)

tf_names = tf_re_motif_df.columns.tolist()
num_TF = len(tf_names)
re_names = tf_re_motif_df.index.tolist()
num_RE = len(re_names)

tf_re_motif_df = tf_re_motif_df.reindex(index=re_expression.columns, columns=tf_expression.columns)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attn_mask = tf_re_motif_df.values
attn_mask_tensor = torch.FloatTensor(attn_mask).to(device)

scaler_tf = StandardScaler()
tf_expression_scaled = scaler_tf.fit_transform(tf_expression.values)

scaler_re = StandardScaler()
re_expression_scaled = scaler_re.fit_transform(re_expression.values)

scaler_tg = StandardScaler()
tg_expression_scaled = scaler_tg.fit_transform(tg_expression.values)
cell_types_df = pd.read_csv(cell_types_file, index_col=0)
cell_types = cell_types_df.iloc[:, 0].values


class TFREDataset(Dataset):
    def __init__(self, tf_expression, re_expression, tg_expression):
        self.tf_expression = torch.FloatTensor(tf_expression)
        self.re_expression = torch.FloatTensor(re_expression)
        self.tg_expression = torch.FloatTensor(tg_expression)
        self.num_samples = self.tf_expression.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'tf_input': self.tf_expression[idx],
            're_input': self.re_expression[idx],
            'tg_target': self.tg_expression[idx],
        }

dataset = TFREDataset(tf_expression_scaled, re_expression_scaled, tg_expression_scaled)

train_indices, val_indices = train_test_split(
    np.arange(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=cell_types
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)



class SparseMultiheadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
        )

        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(query + attn_output)
        return attn_output

class SparseCrossAttentionLayer(nn.Module):

    def __init__(self,
                 embed_dim, num_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="gelu"):
        super().__init__()
        self.cross_attn = SparseMultiheadAttentionLayer(embed_dim, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        activation = activation.lower()
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, query, key, value, attn_mask=None):
        attn_output = self.cross_attn(query, key, value, attn_mask=attn_mask)

        x = self.norm1(query + self.dropout1(attn_output))

        ffn_output = self.linear2(
            self.dropout2( 
                self.activation(self.linear1(x))  
            )
        )


        output = self.norm2(x + ffn_output)  
        return output

class TFREAlignment(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.linear_tf = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_re = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.activation = nn.Tanh()

    def forward(self, tf_embeddings, re_embeddings):

        tf_proj = self.activation(self.linear_tf(tf_embeddings))
        re_proj = self.activation(self.linear_re(re_embeddings))
        scores  = torch.bmm(tf_proj, re_proj.transpose(1, 2))
        return scores

class TFREModel(nn.Module):
    def __init__(self,
                 num_TG, num_RE, num_TF,
                 tg_names, re_names,
                 embedding_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 attn_mask=None,
                 dim_feedforward=2048,
                 activation="gelu",
                 num_layers=1,
                 k_linformer=256):
        super().__init__()
        self.num_TG = num_TG
        self.num_RE = num_RE
        self.num_TF = num_TF

        if attn_mask is not None:
            mask_tf2re = (attn_mask == 0).transpose(0,1).bool()  
            mask_re2tf = (attn_mask == 0).bool()             

            mask_tf2re_float = mask_tf2re.float().masked_fill(mask_tf2re, float('-inf')).masked_fill(~mask_tf2re, 0.0)
            mask_re2tf_float = mask_re2tf.float().masked_fill(mask_re2tf, float('-inf')).masked_fill(~mask_re2tf, 0.0)

            self.register_buffer("attn_mask_tf2re", mask_tf2re_float)  
            self.register_buffer("attn_mask_re2tf", mask_re2tf_float)  
        else:
            self.attn_mask_tf2re = None
            self.attn_mask_re2tf = None

        self.tf_id_embed = nn.Embedding(num_TF, embedding_dim)
        self.re_id_embed = nn.Embedding(num_RE, embedding_dim)
        self.tf_exp_embed = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU()
        )
        self.re_exp_embed = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU()
        )
        self.tf_merge_linear = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU()
        )
        self.re_merge_linear = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU()
        )

        self.cross_layer_tf = SparseCrossAttentionLayer(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.cross_layer_re = SparseCrossAttentionLayer(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        self.transformer_encoder_tf_layer1 = Linformer(
            dim=embedding_dim,
            seq_len=num_TF,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )
        self.transformer_encoder_tf_layer2 = Linformer(
            dim=embedding_dim,
            seq_len=num_TF,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )
        self.transformer_encoder_re_layer1 = Linformer(
            dim=embedding_dim,
            seq_len=num_RE,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )
        self.transformer_encoder_re_layer2 = Linformer(
            dim=embedding_dim,
            seq_len=num_RE,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )

        self.alignment = TFREAlignment(embed_dim=embedding_dim)

        self.combine_attention = Linformer(
            dim=embedding_dim,
            seq_len=(num_TF + num_RE),
            depth=1,
            heads=num_heads,
            k=128,
            one_kv_head=False,
            share_kv=False
        )
        self.mlp_to_tg_1 = nn.Linear(num_TF+num_RE, num_TG)
        self.mlp_to_tg_2 = nn.Linear(embedding_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, tf_input, re_input):

        B = tf_input.size(0)

        tf_idx = torch.arange(self.num_TF, device=tf_input.device).unsqueeze(0).expand(B, -1)
        re_idx = torch.arange(self.num_RE, device=re_input.device).unsqueeze(0).expand(B, -1)

        tf_id_emb = self.tf_id_embed(tf_idx)                  
        tf_exp_emb = self.tf_exp_embed(tf_input.unsqueeze(-1))
        tf_concat = torch.cat([tf_id_emb, tf_exp_emb], dim=-1)
        tf_embedded = self.tf_merge_linear(tf_concat)         

        re_id_emb = self.re_id_embed(re_idx)                  
        re_exp_emb = self.re_exp_embed(re_input.unsqueeze(-1))
        re_concat = torch.cat([re_id_emb, re_exp_emb], dim=-1)
        re_embedded = self.re_merge_linear(re_concat)         

        tf_l1 = self.transformer_encoder_tf_layer1(tf_embedded)  
        re_l1 = self.transformer_encoder_re_layer1(re_embedded)  

        mask_tf2re = getattr(self, "attn_mask_tf2re", None) 

        tf_cross = self.cross_layer_tf(
            query=tf_l1,
            key=re_l1,
            value=re_l1,
            attn_mask=mask_tf2re
        ) 

        mask_re2tf = getattr(self, "attn_mask_re2tf", None)
        re_cross = self.cross_layer_re(
            query=re_l1,
            key=tf_l1,
            value=tf_l1,
            attn_mask=mask_re2tf
        )  

        tf_final = self.transformer_encoder_tf_layer2(tf_cross)
        re_final = self.transformer_encoder_re_layer2(re_cross)

        scores = self.alignment(tf_final, re_final)  
        scores = torch.clamp(scores, min=-10, max=10)

        combined_final = torch.cat([tf_final, re_final], dim=1) 
        combined_final = self.combine_attention(combined_final)  

        tg_pred = combined_final.permute(0, 2, 1)  
        tg_pred = self.mlp_to_tg_1(tg_pred)        
        tg_pred = tg_pred.permute(0, 2, 1)        
        tg_pred = self.relu(tg_pred)
        tg_pred = self.mlp_to_tg_2(tg_pred)       
        tg_pred = tg_pred.squeeze(-1)            

        return tg_pred, scores

num_epochs = 1000  
weight_decay = 5e-6
lr = 1e-5
embedding_dim = 256  
wandb.init(
     
    config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "embedding_dim": embedding_dim,
        "num_heads": 8,
        "dropout": 0.1,
        "dim_feedforward": 2048,
        "activation": "gelu",
        "num_layers": 1,
        "k_linformer": 256
    }
)

model1 = TFREModel(
    num_TG=num_TG,
    num_RE=num_RE,
    num_TF=num_TF,
    tg_names=tg_names,
    re_names=re_names,
    embedding_dim=embedding_dim,
    num_heads=wandb.config["num_heads"],
    dropout=wandb.config["dropout"],
    attn_mask=attn_mask_tensor,
    dim_feedforward=wandb.config["dim_feedforward"],
    activation=wandb.config["activation"],
    num_layers=wandb.config["num_layers"],
    k_linformer=wandb.config["k_linformer"]
).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, Linformer):
        pass
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model1.apply(init_weights)
model1 = nn.DataParallel(model1)

optimizer = AdamW(model1.parameters(), lr=wandb.config["learning_rate"], weight_decay=wandb.config["weight_decay"])

total_steps = len(train_loader) * wandb.config["epochs"]
warmup_steps = int(0.1 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

criterion_tg = nn.MSELoss()
num_pos = attn_mask_tensor.sum().item()
num_total = attn_mask_tensor.numel()
num_neg = num_total - num_pos
pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=attn_mask_tensor.device)
criterion_alignment = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

def get_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m

wandb.watch(model1, log="all")

for epoch in range(wandb.config["epochs"]):
    model1.train()
    total_loss = 0
    total_tg_loss = 0
    total_align_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        tf_input = batch['tf_input'].to(device)
        re_input = batch['re_input'].to(device)
        tg_target = batch['tg_target'].to(device)

        optimizer.zero_grad()

        tg_pred, scores = model1(tf_input, re_input)

        loss_tg = criterion_tg(tg_pred, tg_target)

        motif_matrix_labels = (attn_mask_tensor.transpose(0,1).unsqueeze(0)
                               .expand(tf_input.size(0), -1, -1))
        loss_align = criterion_alignment(scores, motif_matrix_labels)
        loss = loss_tg + 0.5 * loss_align

        loss.backward()
        clip_grad_norm_(model1.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss       += loss.item()
        total_tg_loss    += loss_tg.item()
        total_align_loss += loss_align.item()


    avg_loss = total_loss / len(train_loader)
    avg_tg_loss = total_tg_loss / len(train_loader)
    avg_align_loss = total_align_loss / len(train_loader)

    wandb.log({
        "Loss/Train": avg_loss,
        "TG_Loss/Train": avg_tg_loss,
        "Align_Loss/Train": avg_align_loss,
        "Train/Learning_Rate": scheduler.get_last_lr()[0],
        "epoch": epoch
    })

    model1.eval()
    val_loss = 0
    val_loss_tg = 0
    val_loss_align = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            tf_input = batch['tf_input'].to(device)
            re_input = batch['re_input'].to(device)
            tg_target = batch['tg_target'].to(device)

            tg_pred, scores = model1(tf_input, re_input)
            if tg_pred is None or scores is None:
                continue

            l_tg = criterion_tg(tg_pred, tg_target)
            motif_matrix_labels = (attn_mask_tensor.transpose(0,1).unsqueeze(0)
                                   .expand(tf_input.size(0), -1, -1))
            l_align = criterion_alignment(scores, motif_matrix_labels)
            l_total = l_tg + 0.5*l_align

            val_loss += l_total.item()
            val_loss_tg += l_tg.item()
            val_loss_align += l_align.item()

    if len(val_loader) > 0:
        avg_val_loss = val_loss / len(val_loader)
        avg_val_tg_loss = val_loss_tg / len(val_loader)
        avg_val_align_loss = val_loss_align / len(val_loader)

        wandb.log({
            "Loss/Val": avg_val_loss,
            "TG_Loss/Val": avg_val_tg_loss,
            "Align_Loss/Val": avg_val_align_loss,
            "epoch": epoch
        })

wandb.finish()

if isinstance(model1, torch.nn.DataParallel):
    model1 = model1.module

model1.to(device)
model1.eval()

tf_expression_scaled_df = pd.DataFrame(
    tf_expression_scaled,
    index=tf_expression.index,  
    columns=tf_expression.columns  
)

re_expression_scaled_df = pd.DataFrame(
    re_expression_scaled,
    index=re_expression.index,
    columns=re_expression.columns
)

tg_expression_scaled_df = pd.DataFrame(
    tg_expression_scaled,
    index=tg_expression.index,
    columns=tg_expression.columns
)


tf_expression_all = tf_expression_scaled_df  
re_expression_all = re_expression_scaled_df  
tg_expression_all = tg_expression_scaled_df  

tf_metacell = np.mean(tf_expression_all, axis=0).to_numpy()  
re_metacell = np.mean(re_expression_all, axis=0).to_numpy()  
tg_metacell = np.mean(tg_expression_all, axis=0).to_numpy()  

tf_input_metacell = torch.FloatTensor(tf_metacell).unsqueeze(0).to(device) 
re_input_metacell = torch.FloatTensor(re_metacell).unsqueeze(0).to(device)
tg_target_metacell = torch.FloatTensor(tg_metacell).unsqueeze(0).to(device)

ig_tf = IntegratedGradients(lambda tf: model1(tf, re_input_metacell)[0])

ig_re = IntegratedGradients(lambda re: model1(tf_input_metacell, re)[0])

attributions_tf_list = []

for tg_idx in range(num_TG):
    attributions, delta = ig_tf.attribute(
        inputs=tf_input_metacell,
        target=tg_idx,
        return_convergence_delta=True,
        internal_batch_size=1
    )
    attributions_tf_list.append(attributions.squeeze(0).cpu().detach().numpy())

tf_attributions = np.vstack(attributions_tf_list)

attributions_re_list = []

for tg_idx in range(num_TG):
    attributions, delta = ig_re.attribute(
        inputs=re_input_metacell,
        target=tg_idx,
        return_convergence_delta=True,
        internal_batch_size=1
    )

    attributions_re_list.append(attributions.squeeze(0).cpu().detach().numpy())

re_attributions = np.vstack(attributions_re_list) 

tf_attributions_df = pd.DataFrame(tf_attributions, index=tg_names, columns=tf_names)

tf_attributions_df.to_csv('TFRE_dataenhance_xavier_all_tf_attributions.csv', index=True)
re_attributions_df = pd.DataFrame(re_attributions, index=tg_names, columns=re_names)


re_attributions_df.to_csv('TFRE_dataenhance_xavier_all_re_attributions.csv', index=True)

def experiment1_save_scores_and_alignloss(checkpoint_path, device, tf_input_metacell, re_input_metacell, attn_mask_tensor, save_scores_path="scores_metacell.csv", save_alignloss_path="alignloss_metacell.txt"):
    
    model = model1

    num_pos = attn_mask_tensor.sum().item()
    num_total = attn_mask_tensor.numel()
    num_neg = num_total - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    criterion_alignment = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    motif_matrix_labels_metacell = attn_mask_tensor.transpose(0,1).unsqueeze(0)  
    
    with torch.no_grad():
        tg_pred_metacell, scores_metacell = model(tf_input_metacell, re_input_metacell)
    
    if scores_metacell is not None:

        scores_np = scores_metacell.squeeze(0).cpu().numpy()  
        scores_df = pd.DataFrame(scores_np, index=tf_names, columns=re_names)
        scores_df.to_csv(save_scores_path)

        alignloss = criterion_alignment(scores_metacell, motif_matrix_labels_metacell)


experiment1_save_scores_and_alignloss(
    device=device,
    tf_input_metacell=tf_input_metacell,
    re_input_metacell=re_input_metacell,
    attn_mask_tensor=attn_mask_tensor,
    save_scores_path="TFRE_dataenhance_xavier_scores_metacell.csv",
)


re_expression = pd.read_csv(re_expression_file, index_col=0).T
re_names = re_expression.columns.tolist()
num_RE = len(re_names)


tg_expression = pd.read_csv(tg_expression_file, index_col=0).T 
tg_names = tg_expression.columns.tolist()
num_TG = len(tg_names)


tf_expression = pd.read_csv(tf_expression_file, index_col=0).T 
tf_names = tf_expression.columns.tolist()
num_TF = len(tf_names)

data = np.load(re_tg_matrix_file)
re_tg_sparse_matrix = coo_matrix((data['data'], (data['row'], data['col'])), shape=data['shape'])
re_tg_sparse_matrix = csr_matrix(re_tg_sparse_matrix)  
re_tg_sparse_matrix = pd.DataFrame.sparse.from_spmatrix(re_tg_sparse_matrix, index=tg_names, columns=re_names) 

all_zero_res = re_tg_sparse_matrix.columns[(re_tg_sparse_matrix == 0).all(axis=0)].tolist()
if all_zero_res:
    re_tg_sparse_matrix = re_tg_sparse_matrix.drop(columns=all_zero_res)
    re_expression = re_expression.drop(columns=all_zero_res)


all_zero_tgs = re_tg_sparse_matrix.index[(re_tg_sparse_matrix == 0).all(axis=1)].tolist()
if all_zero_tgs:
    re_tg_sparse_matrix = re_tg_sparse_matrix.drop(index=all_zero_tgs)
    tg_expression = tg_expression.drop(columns=all_zero_tgs)

re_names = re_tg_sparse_matrix.columns.tolist()
num_RE = len(re_names)
tg_names = re_tg_sparse_matrix.index.tolist()
num_TG = len(tg_names)


re_tg_sparse_matrix = re_tg_sparse_matrix.reindex(index=tg_expression.columns, columns=re_expression.columns)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


attn_mask = re_tg_sparse_matrix.values  
attn_mask_tensor = torch.FloatTensor(attn_mask).to(device)  

scaler_re = StandardScaler()
re_expression_scaled = scaler_re.fit_transform(re_expression.values)

scaler_tg = StandardScaler()
tg_expression_scaled = scaler_tg.fit_transform(tg_expression.values)

scaler_tf = StandardScaler()
tf_expression_scaled = scaler_tf.fit_transform(tf_expression.values)

cell_types_df = pd.read_csv(cell_types_file, index_col=0)
cell_types = cell_types_df.iloc[:, 0].values

class RETGDataset(Dataset):
    def __init__(self, re_expression, tg_expression, tf_expression):
        self.re_expression = torch.FloatTensor(re_expression)
        self.tg_expression = torch.FloatTensor(tg_expression)
        self.tf_expression = torch.FloatTensor(tf_expression)
        self.num_samples = self.re_expression.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            're_input': self.re_expression[idx],
            'tg_target': self.tg_expression[idx],
            'tf_input': self.tf_expression[idx],
        }

dataset = RETGDataset(re_expression_scaled, tg_expression_scaled, tf_expression_scaled)

train_indices, val_indices = train_test_split(
    np.arange(len(dataset)),
    test_size=0.2,
    random_state=42,
    stratify=cell_types
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class SparseMultiheadAttentionLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, attn_mask=None):

        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask
        )

        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(query + attn_output)
        return attn_output

class SparseCrossAttentionLayer(nn.Module):
    def __init__(self,
                 embed_dim, num_heads,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="gelu"):
        super().__init__()
        self.cross_attn = SparseMultiheadAttentionLayer(embed_dim, num_heads, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        activation = activation.lower()
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(self, query, key, value, attn_mask=None):
        attn_output = self.cross_attn(query, key, value, attn_mask=attn_mask)

        x = self.norm1(query + self.dropout1(attn_output))

        ffn_output = self.linear2(
            self.dropout2(
                self.activation(self.linear1(x))
            )
        )

        output = self.norm2(x + ffn_output)
        return output

class RETGAlignment(nn.Module):
    def __init__(self, embed_dim, hidden_dim=128):
        super().__init__()
        self.linear_re = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear_tg = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.activation = nn.Tanh()

    def forward(self, re_embeddings, tg_embeddings):

        re_proj = self.activation(self.linear_re(re_embeddings))
        tg_proj = self.activation(self.linear_tg(tg_embeddings))
        scores = torch.bmm(re_proj, tg_proj.transpose(1, 2))
        return scores

class RETGModel(nn.Module):
    def __init__(self,
                 num_TF, num_TG, num_RE,
                 tf_names, tg_names,
                 embedding_dim=256,
                 num_heads=8,
                 dropout=0.1,
                 attn_mask=None,
                 dim_feedforward=2048,
                 activation="gelu",
                 num_layers=1,
                 k_linformer=256):
        super().__init__()
        self.num_TF = num_TF
        self.num_TG = num_TG
        self.num_RE = num_RE

        if attn_mask is not None:
            mask_re2tg = (attn_mask == 0).transpose(0, 1).bool()
            mask_tg2re = (attn_mask == 0).bool()
            
            mask_re2tg_float = mask_re2tg.float().masked_fill(mask_re2tg, float('-inf')).masked_fill(~mask_re2tg, 0.0)
            mask_tg2re_float = mask_tg2re.float().masked_fill(mask_tg2re, float('-inf')).masked_fill(~mask_tg2re, 0.0)

            self.register_buffer("attn_mask_re2tg", mask_re2tg_float) 
            self.register_buffer("attn_mask_tg2re", mask_tg2re_float) 
        else:
            self.attn_mask_re2tg = None
            self.attn_mask_tg2re = None

        self.re_id_embed = nn.Embedding(num_RE, embedding_dim)
        self.tg_id_embed = nn.Embedding(num_TG, embedding_dim)
        self.re_exp_embed = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU()
        )
        self.tg_exp_embed = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU()
        )
        self.re_merge_linear = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU()
        )
        self.tg_merge_linear = nn.Sequential(
            nn.Linear(2*embedding_dim, embedding_dim),
            nn.ReLU()
        )

        self.cross_layer_re = SparseCrossAttentionLayer(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        self.cross_layer_tg = SparseCrossAttentionLayer(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        self.transformer_encoder_re_layer1 = Linformer(
            dim=embedding_dim,
            seq_len=num_RE,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )
        self.transformer_encoder_re_layer2 = Linformer(
            dim=embedding_dim,
            seq_len=num_RE,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )
        self.transformer_encoder_tg_layer1 = Linformer(
            dim=embedding_dim,
            seq_len=num_TG,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )
        self.transformer_encoder_tg_layer2 = Linformer(
            dim=embedding_dim,
            seq_len=num_TG,
            depth=num_layers,
            heads=num_heads,
            k=k_linformer,
            one_kv_head=False,
            share_kv=False
        )

        self.alignment = RETGAlignment(embed_dim=embedding_dim)

        self.combine_attention = Linformer(
            dim=embedding_dim,
            seq_len=(num_RE + num_TG),
            depth=1,
            heads=num_heads,
            k=128,
            one_kv_head=False,
            share_kv=False
        )
        self.mlp_to_tf_1 = nn.Linear(num_RE + num_TG, num_TF)
        self.mlp_to_tf_2 = nn.Linear(embedding_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, re_input, tg_input):

        B = re_input.size(0)

        re_idx = torch.arange(self.num_RE, device=re_input.device).unsqueeze(0).expand(B, -1)
        tg_idx = torch.arange(self.num_TG, device=tg_input.device).unsqueeze(0).expand(B, -1)

        re_id_emb = self.re_id_embed(re_idx)  
        re_exp_emb = self.re_exp_embed(re_input.unsqueeze(-1))  
        re_concat = torch.cat([re_id_emb, re_exp_emb], dim=-1)  
        re_embedded = self.re_merge_linear(re_concat)  
        tg_id_emb = self.tg_id_embed(tg_idx)  
        tg_exp_emb = self.tg_exp_embed(tg_input.unsqueeze(-1))  
        tg_concat = torch.cat([tg_id_emb, tg_exp_emb], dim=-1)  
        tg_embedded = self.tg_merge_linear(tg_concat)  

        re_l1 = self.transformer_encoder_re_layer1(re_embedded)  
        tg_l1 = self.transformer_encoder_tg_layer1(tg_embedded) 

        mask_re2tg = getattr(self, "attn_mask_re2tg", None) 

        re_cross = self.cross_layer_re(
            query=re_l1,
            key=tg_l1,
            value=tg_l1,
            attn_mask=mask_re2tg
        )  

        mask_tg2re = getattr(self, "attn_mask_tg2re", None)
        tg_cross = self.cross_layer_tg(
            query=tg_l1,
            key=re_l1,
            value=re_l1,
            attn_mask=mask_tg2re
        ) 

        re_final = self.transformer_encoder_re_layer2(re_cross)
        tg_final = self.transformer_encoder_tg_layer2(tg_cross)

        scores = self.alignment(re_final, tg_final) 
        scores = torch.clamp(scores, min=-10, max=10)

        combined_final = torch.cat([re_final, tg_final], dim=1) 
        combined_final = self.combine_attention(combined_final)  

        tf_pred = combined_final.permute(0, 2, 1) 
        tf_pred = self.mlp_to_tf_1(tf_pred) 
        tf_pred = tf_pred.permute(0, 2, 1)  
        tf_pred = self.relu(tf_pred)
        tf_pred = self.mlp_to_tf_2(tf_pred) 
        tf_pred = tf_pred.squeeze(-1) 

        return tf_pred, scores

num_epochs = 1000  
weight_decay = 5e-6
lr = 1e-5
embedding_dim = 256  

wandb.init(

    config={
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "embedding_dim": embedding_dim,
        "num_heads": 8,
        "dropout": 0.1,
        "dim_feedforward": 2048,
        "activation": "gelu",
        "num_layers": 1,
        "k_linformer": 256
    }
)

model2 = RETGModel(
    num_TF=num_TF,
    num_TG=num_TG,
    num_RE=num_RE,
    tf_names=tf_names,
    tg_names=tg_names,
    embedding_dim=embedding_dim,
    num_heads=wandb.config["num_heads"],
    dropout=wandb.config["dropout"],
    attn_mask=attn_mask_tensor,
    dim_feedforward=wandb.config["dim_feedforward"],
    activation=wandb.config["activation"],
    num_layers=wandb.config["num_layers"],
    k_linformer=wandb.config["k_linformer"]
).to(device)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, Linformer):
        pass
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model2.apply(init_weights)
model2 = nn.DataParallel(model2)

optimizer = AdamW(model2.parameters(), lr=wandb.config["learning_rate"], weight_decay=wandb.config["weight_decay"])

total_steps = len(train_loader) * wandb.config["epochs"]
warmup_steps = int(0.1 * total_steps) 

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

criterion_tf = nn.MSELoss()
num_pos = attn_mask_tensor.sum().item()
num_total = attn_mask_tensor.numel()
num_neg = num_total - num_pos
pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=attn_mask_tensor.device)
criterion_alignment = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

def get_module(m):
    return m.module if isinstance(m, nn.DataParallel) else m

wandb.watch(model2, log="all")

for epoch in range(wandb.config["epochs"]):
    model2.train()
    total_loss = 0
    total_tf_loss = 0
    total_align_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        re_input = batch['re_input'].to(device)
        tg_input = batch['tg_target'].to(device) 
        tf_target = batch['tf_input'].to(device) 

        optimizer.zero_grad()

        tf_pred, scores = model2(re_input, tg_input)

        loss_tf = criterion_tf(tf_pred, tf_target)

        motif_matrix_labels = (attn_mask_tensor.transpose(0, 1).unsqueeze(0)
                               .expand(re_input.size(0), -1, -1))
        loss_align = criterion_alignment(scores, motif_matrix_labels)
        loss = loss_tf + 0.5 * loss_align

        loss.backward()
        clip_grad_norm_(model2.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  

        total_loss += loss.item()
        total_tf_loss += loss_tf.item()
        total_align_loss += loss_align.item()


    avg_loss = total_loss / len(train_loader)
    avg_tf_loss = total_tf_loss / len(train_loader)
    avg_align_loss = total_align_loss / len(train_loader)

    wandb.log({
        "Loss/Train": avg_loss,
        "TF_Loss/Train": avg_tf_loss,
        "Align_Loss/Train": avg_align_loss,
        "Train/Learning_Rate": scheduler.get_last_lr()[0],
        "epoch": epoch
    })

    model2.eval()
    val_loss = 0
    val_loss_tf = 0
    val_loss_align = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            re_input = batch['re_input'].to(device)
            tg_input = batch['tg_target'].to(device) 
            tf_target = batch['tf_input'].to(device) 

            tf_pred, scores = model2(re_input, tg_input)
            if tf_pred is None or scores is None:
                continue

            l_tf = criterion_tf(tf_pred, tf_target)
            motif_matrix_labels = (attn_mask_tensor.transpose(0, 1).unsqueeze(0)
                                   .expand(re_input.size(0), -1, -1))
            l_align = criterion_alignment(scores, motif_matrix_labels)
            l_total = l_tf + 0.5*l_align

            val_loss += l_total.item()
            val_loss_tf += l_tf.item()
            val_loss_align += l_align.item()

    if len(val_loader) > 0:
        avg_val_loss = val_loss / len(val_loader)
        avg_val_tf_loss = val_loss_tf / len(val_loader)
        avg_val_align_loss = val_loss_align / len(val_loader)

        wandb.log({
            "Loss/Val": avg_val_loss,
            "TF_Loss/Val": avg_val_tf_loss,
            "Align_Loss/Val": avg_val_align_loss,
            "epoch": epoch
        })

wandb.finish()

if isinstance(model2, torch.nn.DataParallel):
    model2 = model2.module

model2.to(device)
model2.eval()

re_expression_scaled_df = pd.DataFrame(
    re_expression_scaled,
    index=re_expression.index, 
    columns=re_expression.columns 
)

tg_expression_scaled_df = pd.DataFrame(
    tg_expression_scaled,
    index=tg_expression.index,
    columns=tg_expression.columns
)

tf_expression_scaled_df = pd.DataFrame(
    tf_expression_scaled,
    index=tf_expression.index,
    columns=tf_expression.columns
)


re_expression_all = re_expression_scaled_df  
tg_expression_all = tg_expression_scaled_df 
tf_expression_all = tf_expression_scaled_df 

re_metacell = np.mean(re_expression_all, axis=0).to_numpy()  
tg_metacell = np.mean(tg_expression_all, axis=0).to_numpy()  
tf_metacell = np.mean(tf_expression_all, axis=0).to_numpy() 

re_input_metacell = torch.FloatTensor(re_metacell).unsqueeze(0).to(device) 
tg_input_metacell = torch.FloatTensor(tg_metacell).unsqueeze(0).to(device)  
tf_target_metacell = torch.FloatTensor(tf_metacell).unsqueeze(0).to(device) 

ig_re = IntegratedGradients(lambda re: model2(re, tg_input_metacell)[0])

ig_tg = IntegratedGradients(lambda tg: model2(re_input_metacell, tg)[0])

attributions_re_list = []

for tf_idx in range(num_TF):
    attributions, delta = ig_re.attribute(
        inputs=re_input_metacell,
        target=tf_idx,
        return_convergence_delta=True,
        internal_batch_size=1
    )

    attributions_re_list.append(attributions.squeeze(0).cpu().detach().numpy())

re_attributions = np.vstack(attributions_re_list)

attributions_tg_list = []

for tf_idx in range(num_TF):
    attributions, delta = ig_tg.attribute(
        inputs=tg_input_metacell,
        target=tf_idx,
        return_convergence_delta=True,
        internal_batch_size=1
    )
    attributions_tg_list.append(attributions.squeeze(0).cpu().detach().numpy())

tg_attributions = np.vstack(attributions_tg_list)

re_attributions_df = pd.DataFrame(re_attributions, index=tf_names, columns=re_names)

re_attributions_df.to_csv('RETG_dataenhance_xavier_all_re_attributions.csv', index=True)

tg_attributions_df = pd.DataFrame(tg_attributions, index=tf_names, columns=tg_names)


tg_attributions_df.to_csv('RETG_dataenhance_xavier_all_tg_attributions.csv', index=True)

def experiment1_save_scores_and_alignloss(checkpoint_path, device, tg_input_metacell, re_input_metacell, attn_mask_tensor, save_scores_path="scores_metacell.csv"):

    model = model2

    if next(iter(model_state_dict)).startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
    model.to(device)
    model.eval() 

    num_pos = attn_mask_tensor.sum().item()
    num_total = attn_mask_tensor.numel()
    num_neg = num_total - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    criterion_alignment = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    motif_matrix_labels_metacell = attn_mask_tensor.transpose(0,1).unsqueeze(0)
    
    with torch.no_grad():
        tf_pred_metacell, scores_metacell = model(re_input_metacell, tg_input_metacell)

    if scores_metacell is not None:

        scores_np = scores_metacell.squeeze(0).cpu().numpy() 
        scores_df = pd.DataFrame(scores_np, index=re_names, columns=tg_names)
        scores_df.to_csv(save_scores_path)

        alignloss = criterion_alignment(scores_metacell, motif_matrix_labels_metacell)

experiment1_save_scores_and_alignloss(
    device=device,
    tg_input_metacell=tg_input_metacell,
    re_input_metacell=re_input_metacell,
    attn_mask_tensor=attn_mask_tensor,
    save_scores_path="RETG_dataenhance_xavier_scores_metacell.csv",
)
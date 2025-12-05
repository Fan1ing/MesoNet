import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import GCNConv, Set2Set, global_mean_pool, NNConv
from torch_geometric.utils import subgraph as pyg_subgraph
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from torch.utils.data import Subset
from pathlib import Path

from data_processing import *
from model import *




epochs = 180
k_folds = 5
batch_size =  256
input_dim = atom_featurizer.dim
edge_dim = bond_featurizer.dim
hidden_dim = 160
output_dim = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
threshold = 400
dataset_size = len(dataset)
kf = KFold(n_splits=k_folds, shuffle=True, random_state=2021)
start_fold = 0
best_val_losses, best_val_maes, best_val_mses, best_val_r2s = [], [], [], []
test_rmse_list, test_mae_list, test_mse_list, test_r2_list = [], [], [], []

for fold, (train_idx, valtest_idx) in enumerate(kf.split(dataset)):
    if fold < start_fold:
        print(f"Skipping Fold {fold+1}")
        continue

    print(f"Start Fold {fold+1}/{k_folds}")

    val_idx, test_idx = train_test_split(
        valtest_idx, test_size=0.5, random_state=42, shuffle=True
    )

    train_subset = [dataset[i] for i in train_idx]
    val_subset = [dataset[i] for i in val_idx]
    test_subset = [dataset[i] for i in test_idx]
    print(f"Fold {fold+1} ：")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")
    print(f"  total:  {len(train_idx) + len(val_idx) + len(test_idx)}\n")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model = MesoNet(input_dim, edge_dim, hidden_dim=160, output_dim=1,
                    d_group_in=160).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()

    best_val_rmse = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(epochs):

        model.train()
        y_train_true, y_train_pred = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            output, _,attn_info = model(batch)
            output = output.view(-1, 1)
            target = batch.y.unsqueeze(1).to(device)
            mask = torch.abs(target) < threshold

            output = output[mask]
            target = target[mask]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            y_train_true.extend(target.cpu().numpy().flatten())
            y_train_pred.extend(output.detach().cpu().numpy().flatten())

        train_mse = mean_squared_error(y_train_true, y_train_pred)
        train_rmse = math.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train_true, y_train_pred)
        train_r2 = r2_score(y_train_true, y_train_pred)

        # ---- validation ----
        model.eval()
        y_val_true, y_val_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output, _,_ = model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)
                mask = torch.abs(target) < threshold

                output = output[mask]
                target = target[mask]
                y_val_true.extend(target.cpu().numpy().flatten())
                y_val_pred.extend(output.cpu().numpy().flatten())

        val_mse = mean_squared_error(y_val_true, y_val_pred)
        val_rmse = math.sqrt(val_mse)
        val_mae = mean_absolute_error(y_val_true, y_val_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)
        y_test_true, y_test_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output, _ ,_= model(batch)
                output = output.view(-1, 1)
                target = batch.y.unsqueeze(1).to(device)
                mask = torch.abs(target) < threshold
                output = output[mask]
                target = target[mask]
                y_test_true.extend(target.cpu().numpy().flatten())
                y_test_pred.extend(output.cpu().numpy().flatten())
        test_mse = mean_squared_error(y_test_true, y_test_pred)
        test_rmse = math.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test_true, y_test_pred)
        test_r2 = r2_score(y_test_true, y_test_pred)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            bsettest_mae, bsettest_rmse, bsettest_r2 = test_mae, test_rmse, test_r2
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"  Val   RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    test_rmse_list.append(bsettest_rmse)
    test_mae_list.append(bsettest_mae)
    test_r2_list.append(bsettest_r2)

    print(f"\nFold {fold+1} Best Epoch {best_epoch}")
    print(f"  Val RMSE: {best_val_rmse:.4f}, Test RMSE: {bsettest_rmse:.4f}, Test MAE: {bsettest_mae:.4f}, Test R²: {bsettest_r2:.4f}")

print("\nAverage Results Across Folds:")
print(f"  Avg Test RMSE: {np.mean(test_rmse_list):.4f}, Avg Test MAE: {np.mean(test_mae_list):.4f}, Avg Test R²: {np.mean(test_r2_list):.4f}")

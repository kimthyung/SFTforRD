"""
Train BiLSTM rack-force regressor.

  1. Build train/val splits per scenario (5:1 split, no shuffling).
  2. Fit StandardScalers on the training portion only.
  3. Train BiLSTM with early stopping (val MSE patience).
  4. Save weights to `weights/BiLSTM.pt`.

Usage:
    cd src && python train_bilstm.py
"""

import os
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_COLS, TARGET_COL,
    TRAIN_FILES,
    SEQ_LEN, BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE, TRAIN_RATIO,
    BILSTM_HIDDEN, BILSTM_LAYERS, SEED,
    WEIGHT_DIR,
)
from models import BiLSTMRegressor
from data import load_csv, create_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(WEIGHT_DIR, exist_ok=True)


def prepare_loaders():
    """Per-scenario contiguous split; fit scalers on the TRAIN portion only."""
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for path in TRAIN_FILES:
        df = load_csv(path)
        X = df[INPUT_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
        n_tr = int(TRAIN_RATIO * len(X))
        tr_x.append(X[:n_tr]); tr_y.append(y[:n_tr])
        va_x.append(X[n_tr:]); va_y.append(y[n_tr:])
        print(f"  {os.path.basename(path):42s}: total={len(X):6d} "
              f"train={n_tr:6d} val={len(X)-n_tr:5d}")

    sx, sy = StandardScaler(), StandardScaler()
    sx.fit(np.concatenate(tr_x))
    sy.fit(np.concatenate(tr_y))

    def to_seqs(raw_X, raw_y):
        axs, ays = [], []
        for rx, ry in zip(raw_X, raw_y):
            if len(rx) < SEQ_LEN:
                continue
            xs, ys = create_sequences(sx.transform(rx), sy.transform(ry), SEQ_LEN)
            axs.append(xs); ays.append(ys)
        return np.concatenate(axs), np.concatenate(ays)

    xt, yt = to_seqs(tr_x, tr_y)
    xv, yv = to_seqs(va_x, va_y)
    print(f"  Train sequences: {len(xt)},  Val sequences: {len(xv)}")

    train_dl = DataLoader(
        TensorDataset(torch.tensor(xt), torch.tensor(yt)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(xv), torch.tensor(yv)),
        batch_size=BATCH_SIZE,
    )
    return train_dl, val_dl


def train():
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = BiLSTMRegressor(len(INPUT_COLS), BILSTM_HIDDEN, BILSTM_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dl, val_dl = prepare_loaders()

    best_val, counter, best_state, stop_ep = float('inf'), 0, None, EPOCHS
    for epoch in range(EPOCHS):
        model.train(); tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tr_loss += loss.item()

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_loss += criterion(model(xb), yb).item()
        av = va_loss / len(val_dl)

        if av < best_val - 1e-3:
            best_val, counter = av, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
        if counter >= PATIENCE:
            stop_ep = epoch + 1
            break
        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch+1:4d}: train={tr_loss/len(train_dl):.6f}  "
                  f"val={av:.6f}")

    model.load_state_dict(best_state)
    wt = os.path.join(WEIGHT_DIR, 'BiLSTM.pt')
    torch.save(model.state_dict(), wt)
    print(f"  stopped@{stop_ep}, best_val={best_val:.6f}, params={n_params:,d}")
    print(f"  saved: {wt}")


def main():
    print(f"Device : {device}")
    print(f"BiLSTM : h={BILSTM_HIDDEN}, L={BILSTM_LAYERS}")
    print(f"Seed   : {SEED}")
    print()
    t0 = time.time()
    train()
    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()

"""
Data loading, scaling, and sequencing utilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import TARGET_COL


def load_csv(path):
    """
    Load a CarMaker-style CSV.

    Handles the legacy '#Name' header convention by shifting columns,
    drops auxiliary book-keeping columns (`source_*`), and synthesizes
    the target rack force from left+right tie-rod forces.
    """
    df = pd.read_csv(path)

    if df.columns[0] == '#Name':
        df.columns = list(df.columns[1:]) + ['_drop']
        df = df.drop(columns=['_drop'])
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all')

    for c in ['source_file', 'source_row_start', 'source_row_end']:
        if c in df.columns:
            df = df.drop(columns=[c])

    df[TARGET_COL] = df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2']
    return df


def create_sequences(X, y, seq_len):
    """Return (xs, ys) where xs[i] = X[i:i+seq_len] and ys[i] = y[i+seq_len-1]."""
    n = len(X) - seq_len + 1
    if n <= 0:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), \
               np.zeros((0, 1), dtype=np.float32)
    xs = np.stack([X[i:i + seq_len] for i in range(n)], axis=0)
    ys = y[seq_len - 1: seq_len - 1 + n]
    return xs.astype(np.float32), ys.astype(np.float32)


def fit_scalers(train_files, input_cols, train_ratio):
    """Fit StandardScalers on the training portion of each training file."""
    rx, ry = [], []
    for path in train_files:
        df = load_csv(path)
        n = int(train_ratio * len(df))
        rx.append(df[input_cols].values[:n].astype(np.float32))
        ry.append(df[TARGET_COL].values[:n].astype(np.float32).reshape(-1, 1))
    sx, sy = StandardScaler(), StandardScaler()
    sx.fit(np.concatenate(rx))
    sy.fit(np.concatenate(ry))
    return sx, sy

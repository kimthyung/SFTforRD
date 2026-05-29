"""
Fit ELM residual corrector on top of a trained BiLSTM.

  1. Run BiLSTM (weights/BiLSTM.pt) over the slip-rich files in
     `data/elm_train/` to compute predictions.
  2. Compute residual r = actual − BiLSTM_pred (in original units).
  3. Fit an ELM(input=5 ELM features, hidden, activation, C) that maps the
     same time-aligned features to the residual.
  4. Save the ELM (W, b, beta, hyperparams) to `weights/ELM.pt`.

Usage:
    cd src && python fit_elm.py
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_COLS, ELM_COLS, TARGET_COL,
    TRAIN_FILES, ELM_FIT_FILES,
    SEQ_LEN, BATCH_SIZE, TRAIN_RATIO,
    BILSTM_HIDDEN, BILSTM_LAYERS,
    ELM_HIDDEN, ELM_ACTIVATION, ELM_C, SEED,
    WEIGHT_DIR,
)
from models import BiLSTMRegressor, ELM
from data import load_csv, create_sequences, fit_scalers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bilstm_predict_unscaled(model, sx, sy, df):
    """Run BiLSTM on `df` and return (pred, actual) in original units."""
    X = sx.transform(df[INPUT_COLS].values.astype(np.float32))
    y = sy.transform(df[TARGET_COL].values.astype(np.float32).reshape(-1, 1))
    xq, yq = create_sequences(X, y, SEQ_LEN)
    with torch.no_grad():
        preds = []
        for i in range(0, len(xq), BATCH_SIZE):
            preds.append(model(torch.tensor(xq[i:i+BATCH_SIZE]).to(device))
                         .cpu().numpy())
    pred = sy.inverse_transform(np.concatenate(preds)).flatten()
    actual = sy.inverse_transform(yq).flatten()
    return pred, actual


def main():
    print(f"Device : {device}")
    print(f"BiLSTM : h={BILSTM_HIDDEN}, L={BILSTM_LAYERS}")
    print(f"ELM    : hidden={ELM_HIDDEN}, act={ELM_ACTIVATION}, C={ELM_C}")
    print()

    sx, sy = fit_scalers(TRAIN_FILES, INPUT_COLS, TRAIN_RATIO)

    wt = os.path.join(WEIGHT_DIR, 'BiLSTM.pt')
    if not os.path.exists(wt):
        raise FileNotFoundError(f"BiLSTM weights not found: {wt}")

    model = BiLSTMRegressor(len(INPUT_COLS), BILSTM_HIDDEN, BILSTM_LAYERS).to(device)
    model.load_state_dict(torch.load(wt, map_location=device))
    model.eval()

    X_list, y_list = [], []
    for fit_file in ELM_FIT_FILES:
        df = load_csv(fit_file)
        pred, actual = bilstm_predict_unscaled(model, sx, sy, df)
        residual = actual - pred
        off = SEQ_LEN - 1
        inp = df[ELM_COLS].values[off:off + len(pred)].astype(np.float64)
        X_list.append(inp)
        y_list.append(residual)
        print(f"  {os.path.basename(fit_file):30s} samples={len(pred):6d}  "
              f"|residual|≈{np.mean(np.abs(residual)):.2f}")

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list).reshape(-1, 1)

    elm = ELM(
        input_size=len(ELM_COLS),
        hidden_size=ELM_HIDDEN,
        activation=ELM_ACTIVATION,
        C=ELM_C,
        seed=SEED,
    )
    elm.fit(torch.from_numpy(X_all), torch.from_numpy(y_all))

    out = os.path.join(WEIGHT_DIR, 'ELM.pt')
    torch.save(elm.state_dict(), out)
    print(f"\nELM fit on {len(X_all)} samples → saved {out}")


if __name__ == '__main__':
    main()

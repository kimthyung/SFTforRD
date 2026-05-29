"""
Combined BiLSTM + ELM inference on a single CSV file.

  • BiLSTM produces a base rack-force prediction from the 5-feature input
    sequence.
  • For samples whose vehicle state satisfies the slip mask
      vx >= VX_THRESH AND (|slip_angle| >= SLIP_THRESH OR |long_slip| >= LONG_THRESH)
    the ELM residual correction is added.

Usage:
    cd src && python predict.py --input ../data/test/Test_set_gm_mu_0.4.CSV
    cd src && python predict.py --input ../data/test/Test_set_gm_mu_0.4.CSV --output preds.npz
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_COLS, ELM_COLS, TARGET_COL,
    TRAIN_FILES,
    SEQ_LEN, BATCH_SIZE, TRAIN_RATIO,
    BILSTM_HIDDEN, BILSTM_LAYERS,
    SLIP_THRESH, LONG_THRESH, VX_THRESH,
    WEIGHT_DIR,
)
from models import BiLSTMRegressor, ELM
from data import load_csv, create_sequences, fit_scalers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SLIP_REQUIRED_COLS = [
    'Car.SlipAngleFL', 'Car.SlipAngleFR',
    'Car.LongSlipFL', 'Car.LongSlipFR', 'Car.vx',
]


def load_pipeline():
    """Load scaler, BiLSTM, and ELM."""
    sx, sy = fit_scalers(TRAIN_FILES, INPUT_COLS, TRAIN_RATIO)

    bilstm = BiLSTMRegressor(len(INPUT_COLS), BILSTM_HIDDEN, BILSTM_LAYERS).to(device)
    bilstm.load_state_dict(
        torch.load(os.path.join(WEIGHT_DIR, 'BiLSTM.pt'), map_location=device)
    )
    bilstm.eval()

    elm = ELM.from_state_dict(
        torch.load(os.path.join(WEIGHT_DIR, 'ELM.pt'), map_location='cpu')
    )
    return sx, sy, bilstm, elm


def predict(df, sx, sy, bilstm, elm):
    """
    Return dict with bilstm_pred, combined_pred, mask, (optionally) actual.
    All predictions are in original units (Newton).
    """
    X = sx.transform(df[INPUT_COLS].values.astype(np.float32))
    xq, _ = create_sequences(
        X, np.zeros((len(X), 1), dtype=np.float32), SEQ_LEN,
    )

    with torch.no_grad():
        preds_sc = []
        for i in range(0, len(xq), BATCH_SIZE):
            preds_sc.append(
                bilstm(torch.tensor(xq[i:i+BATCH_SIZE]).to(device)).cpu().numpy()
            )
    bilstm_pred = sy.inverse_transform(np.concatenate(preds_sc)).flatten()

    off = SEQ_LEN - 1
    n_pred = len(bilstm_pred)
    combined = bilstm_pred.copy()

    if all(c in df.columns for c in SLIP_REQUIRED_COLS):
        sa = (df['Car.SlipAngleFL'].values + df['Car.SlipAngleFR'].values) / 2
        ls = (df['Car.LongSlipFL'].values + df['Car.LongSlipFR'].values) / 2
        vx = df['Car.vx'].values
        sa_w = sa[off:off + n_pred]
        ls_w = ls[off:off + n_pred]
        vx_w = vx[off:off + n_pred]
        mask = (vx_w >= VX_THRESH) & (
            (np.abs(sa_w) >= SLIP_THRESH) | (np.abs(ls_w) >= LONG_THRESH)
        )
        elm_inp = df[ELM_COLS].values[off:off + n_pred].astype(np.float64)
        corr = elm.predict(torch.from_numpy(elm_inp)).numpy().flatten()
        combined[mask] = combined[mask] + corr[mask]
    else:
        mask = np.zeros(n_pred, dtype=bool)

    actual = None
    if TARGET_COL in df.columns or {'Car.CFL.GenFrc2', 'Car.CFR.GenFrc2'}.issubset(df.columns):
        if TARGET_COL not in df.columns:
            df = df.copy()
            df[TARGET_COL] = df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2']
        actual = df[TARGET_COL].values.astype(np.float64)[off:off + n_pred]

    return {
        'bilstm_pred':   bilstm_pred,
        'combined_pred': combined,
        'mask':          mask,
        'actual':        actual,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Path to input CSV.')
    parser.add_argument('--output', default=None,
                        help='Optional path to save predictions as .npz.')
    args = parser.parse_args()

    print(f"Device : {device}")
    print(f"Input  : {args.input}")

    sx, sy, bilstm, elm = load_pipeline()
    df = load_csv(args.input)
    out = predict(df, sx, sy, bilstm, elm)

    n  = len(out['bilstm_pred'])
    nm = int(out['mask'].sum())
    print(f"Samples (after windowing): {n}")
    print(f"ELM-corrected samples    : {nm}  ({100*nm/max(n,1):.1f}%)")

    if out['actual'] is not None:
        a  = out['actual']
        rb = np.sqrt(np.mean((out['bilstm_pred']   - a) ** 2))
        rc = np.sqrt(np.mean((out['combined_pred'] - a) ** 2))
        print(f"RMSE BiLSTM   : {rb:8.2f} N")
        print(f"RMSE Combined : {rc:8.2f} N")

    if args.output:
        np.savez(args.output, **out)
        print(f"Saved predictions: {args.output}")


if __name__ == '__main__':
    main()

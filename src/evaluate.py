"""
Evaluate the full BiLSTM + ELM pipeline on the 8 test scenarios.

NRMSE is normalized by the per-scenario max−min of the ground-truth rack
force (values in `config.TEST_FILES`).

Usage:
    cd src && python evaluate.py
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    INPUT_COLS, ELM_COLS, TARGET_COL,
    TRAIN_FILES, TEST_FILES, TEST_DIR,
    SEQ_LEN, BATCH_SIZE, TRAIN_RATIO,
    BILSTM_HIDDEN, BILSTM_LAYERS,
    ELM_HIDDEN, ELM_ACTIVATION, ELM_C,
    SLIP_THRESH, LONG_THRESH, VX_THRESH,
    WEIGHT_DIR, RESULT_DIR,
)
from models import BiLSTMRegressor, ELM
from data import load_csv, create_sequences, fit_scalers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(RESULT_DIR, exist_ok=True)


def evaluate_bilstm_elm(bilstm, elm, sx, sy, df):
    X = sx.transform(df[INPUT_COLS].values.astype(np.float32))
    y = sy.transform(df[TARGET_COL].values.astype(np.float32).reshape(-1, 1))
    xq, yq = create_sequences(X, y, SEQ_LEN)
    if len(xq) == 0:
        return None, None

    with torch.no_grad():
        chunks = []
        for i in range(0, len(xq), BATCH_SIZE):
            chunks.append(
                bilstm(torch.tensor(xq[i:i+BATCH_SIZE]).to(device)).cpu().numpy()
            )
    pred   = sy.inverse_transform(np.concatenate(chunks)).flatten()
    actual = sy.inverse_transform(yq).flatten()

    combined = pred.copy()
    if all(c in df.columns for c in
           ['Car.SlipAngleFL', 'Car.SlipAngleFR',
            'Car.LongSlipFL', 'Car.LongSlipFR', 'Car.vx']):
        off = SEQ_LEN - 1
        n = len(pred)
        sa = (df['Car.SlipAngleFL'].values + df['Car.SlipAngleFR'].values) / 2
        ls = (df['Car.LongSlipFL'].values + df['Car.LongSlipFR'].values) / 2
        vx = df['Car.vx'].values
        sa_w, ls_w, vx_w = sa[off:off+n], ls[off:off+n], vx[off:off+n]
        mask = (vx_w >= VX_THRESH) & (
            (np.abs(sa_w) >= SLIP_THRESH) | (np.abs(ls_w) >= LONG_THRESH)
        )
        elm_inp = df[ELM_COLS].values[off:off+n].astype(np.float64)
        corr = elm.predict(torch.from_numpy(elm_inp)).numpy().flatten()
        combined[mask] = combined[mask] + corr[mask]

    rb = float(np.sqrt(np.mean((pred     - actual) ** 2)))
    rc = float(np.sqrt(np.mean((combined - actual) ** 2)))
    return rb, rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
                        default=os.path.join(RESULT_DIR, 'evaluation.csv'))
    args = parser.parse_args()

    print(f"Device : {device}")
    print(f"BiLSTM : h={BILSTM_HIDDEN}, L={BILSTM_LAYERS}")
    print(f"ELM    : hidden={ELM_HIDDEN}, act={ELM_ACTIVATION}, C={ELM_C}")
    print()

    sx, sy = fit_scalers(TRAIN_FILES, INPUT_COLS, TRAIN_RATIO)

    bilstm = BiLSTMRegressor(len(INPUT_COLS), BILSTM_HIDDEN, BILSTM_LAYERS).to(device)
    bilstm.load_state_dict(
        torch.load(os.path.join(WEIGHT_DIR, 'BiLSTM.pt'), map_location=device)
    )
    bilstm.eval()
    elm = ELM.from_state_dict(
        torch.load(os.path.join(WEIGHT_DIR, 'ELM.pt'), map_location='cpu')
    )

    rows = []
    for tag, (filename, rng) in TEST_FILES.items():
        path = os.path.join(TEST_DIR, filename)
        df = load_csv(path)
        rb, rc = evaluate_bilstm_elm(bilstm, elm, sx, sy, df)
        if rb is None:
            continue
        rows.append({
            'test_set':       tag,
            'range':          rng,
            'rmse_bilstm':    rb,
            'nrmse_bilstm':   rb / rng * 100,
            'rmse_combined':  rc,
            'nrmse_combined': rc / rng * 100,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.output, index=False)

    print(f"{'test_set':>22s}   {'BiLSTM':>8s}   {'+ELM':>8s}")
    for _, r in df_out.iterrows():
        print(f"  {r['test_set']:>22s}   "
              f"{r['nrmse_bilstm']:>7.2f}%   {r['nrmse_combined']:>7.2f}%")
    print(f"  {'OVERALL':>22s}   "
          f"{df_out['nrmse_bilstm'].mean():>7.2f}%   "
          f"{df_out['nrmse_combined'].mean():>7.2f}%")
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()

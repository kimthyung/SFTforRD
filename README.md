# Rack Force Estimation with BiLSTM + ELM

Reference implementation of a hybrid **BiLSTM + Extreme Learning Machine
(ELM)** model that estimates the rack force of an electric power steering
system from five vehicle-dynamics signals available at the remote driver.

The BiLSTM produces a base estimate of the rack force from a short window
of past samples; the ELM learns a closed-form residual corrector that is
applied only on slip-dominant samples (low-µ, large slip angle, etc.).

## Repository layout

```
RackForce_BiLSTM_ELM/
├── src/
│   ├── config.py              # all hyperparameters and paths (edit this!)
│   ├── models.py              # BiLSTMRegressor, ELM
│   ├── data.py                # CSV loading, scaling, sequencing
│   ├── train_bilstm.py        # train BiLSTM
│   ├── fit_elm.py             # fit ELM residual corrector
│   ├── predict.py             # single-file inference
│   ├── evaluate.py            # benchmark on the 8 test scenarios
│   └── baselines/
│       └── nn_baselines.py    # BiGRURegressor, LSTMRegressor
├── data/
│   ├── train/                 # 4 training CSVs for the BiLSTM
│   ├── elm_train/             # 3 slip-rich CSVs used to fit the ELM
│   └── test/                  # 8 evaluation CSVs
├── weights/
│   ├── BiLSTM.pt              # trained BiLSTM
│   ├── ELM.pt                 # fitted ELM
│   └── baselines/
│       ├── BiGRU.pt           # baseline BiGRU
│       └── LSTM.pt            # baseline LSTM
├── results/                   # evaluation.csv (created by evaluate.py)
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code runs on either CPU or CUDA; CUDA is auto-detected by `torch`.

## Pipeline overview

The proposed system is a **two-stage cascade**:

1. **BiLSTM (base regressor)** — A stacked, bidirectional LSTM is trained
   on the four `Train_set_*.CSV` files to map a 10-sample window of 5
   vehicle-dynamics signals to the (scaled) rack force.

2. **ELM residual corrector** — Once the BiLSTM is frozen, an ELM is fit
   in closed form to predict the residual
   `actual − BiLSTM_pred` from 5 slip-aware features. The ELM is fit on
   three slip-rich datasets stored under `data/elm_train/`.

At inference time the ELM correction is **applied only when the slip
mask is active**:

```
mask = (vx >= VX_THRESH) & (|slip_angle| >= SLIP_THRESH OR
                            |long_slip|  >= LONG_THRESH)
```

so that low-speed, straight-line, and stationary-steering samples are
not perturbed by the ELM.

## Model configuration

All hyperparameters live in [`src/config.py`](src/config.py). The
release-tested defaults are:

| Component | Setting | Value |
|:--|:--|:--|
| BiLSTM | input features | `Car.YawRate`, `Car.ay`, `Driver.Steer.Ang`, `Driver.Steer.AngVel`, `Car.vx` |
| BiLSTM | hidden size / layers | 16 / 3 (bidirectional) |
| BiLSTM | sequence length | 10 (100 ms @ 100 Hz) |
| BiLSTM | optimizer | Adam, lr = 1e-3, batch = 128 |
| BiLSTM | early stopping | patience = 200 epochs |
| BiLSTM | training data | `data/train/Train_set_1..4.CSV` |
| ELM | input features | `Car.SlipAngleFL`, `Car.SlipAngleFR`, `Car.YawRate`, `Driver.Steer.Ang`, `Car.ay` |
| ELM | hidden units | 100 |
| ELM | activation | sigmoid |
| ELM | ridge regularization C | 0.1 |
| ELM | fitting data | `data/elm_train/Test_set_{L14_H, L16_H, L20_H}.CSV` |
| Slip mask | `VX_THRESH` (m/s) | 10 / 3.6  (≈ 10 km/h) |
| Slip mask | `SLIP_THRESH` (rad) | 0.025 |
| Slip mask | `LONG_THRESH` (-)  | 0.003 |

To experiment with a different configuration, edit `config.py` once;
every script reads its settings from there.

## Usage

All commands assume you are in `src/`:

```bash
cd src
```

### 1. (Optional) Train the BiLSTM from scratch

A pre-trained `BiLSTM.pt` ships under `weights/`. Re-training takes a
few minutes on a single GPU; on CPU it is considerably slower.

```bash
python train_bilstm.py
```

Output: `weights/BiLSTM.pt`.

### 2. Fit the ELM residual corrector

```bash
python fit_elm.py
```

The script

1. runs the BiLSTM over the three slip-rich CSVs in `data/elm_train/`,
2. computes the residual `actual − BiLSTM_pred` in Newtons,
3. fits an `ELM(ELM_HIDDEN, ELM_ACTIVATION, C=ELM_C)` on the
   `ELM_COLS` features → residual map,
4. saves `weights/ELM.pt`.

ELM fitting is a closed-form ridge solve and finishes in seconds.

### 3. Run inference on a single CSV

```bash
python predict.py --input ../data/test/Test_set_gm_mu_0.4.CSV
python predict.py --input ../data/test/Test_set_gm_mu_0.4.CSV --output preds.npz
```

The script reports

* the number of samples (after windowing),
* the percentage of samples the slip mask selected for ELM correction,
* RMSE / NRMSE for BiLSTM-only and BiLSTM + ELM (only printed if the
  CSV contains ground-truth rack-force columns).

When `--output preds.npz` is supplied, the file contains
`bilstm_pred`, `combined_pred`, `mask`, and (when available) `actual`.

### 4. Benchmark on the 8 test scenarios

```bash
python evaluate.py                 # BiLSTM and BiLSTM + ELM only

```


"""
Configuration for BiLSTM + ELM Rack Force Estimation.

All model and pipeline hyperparameters live here.
Edit this file to change settings; do not hard-code values in other modules.
"""

import os


# ──────────────────────────────────────────────────────────────────────
# Paths (resolved relative to repository root)
# ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR      = os.path.join(ROOT_DIR, 'data')
TRAIN_DIR     = os.path.join(DATA_DIR, 'train')
TEST_DIR      = os.path.join(DATA_DIR, 'test')
ELM_TRAIN_DIR = os.path.join(DATA_DIR, 'elm_train')
WEIGHT_DIR    = os.path.join(ROOT_DIR, 'weights')
RESULT_DIR    = os.path.join(ROOT_DIR, 'results')


# ──────────────────────────────────────────────────────────────────────
# Feature columns
# ──────────────────────────────────────────────────────────────────────
# BiLSTM inputs (5 vehicle-dynamics signals available at the remote driver)
INPUT_COLS = [
    'Car.YawRate',
    'Car.ay',
    'Driver.Steer.Ang',
    'Driver.Steer.AngVel',
    'Car.vx',
]

# Target = sum of left and right tie-rod forces
TARGET_COL = 'RackForce'

# ELM inputs (5 features chosen for residual correction on slip events)
ELM_COLS = [
    'Car.SlipAngleFL',
    'Car.SlipAngleFR',
    'Car.YawRate',
    'Driver.Steer.Ang',
    'Car.ay',
]


# ──────────────────────────────────────────────────────────────────────
# BiLSTM hyperparameters
# ──────────────────────────────────────────────────────────────────────
SEQ_LEN       = 10          # input sequence length (samples @ 100 Hz = 100 ms)
BILSTM_HIDDEN = 16          # LSTM hidden size per direction
BILSTM_LAYERS = 3           # number of stacked LSTM layers

BATCH_SIZE    = 128
LEARNING_RATE = 1e-3
EPOCHS        = 1000
PATIENCE      = 200         # early stopping patience (epochs)
TRAIN_RATIO   = 5 / 6       # per-scenario train/val split
SEED          = 43          # random seed used for both BiLSTM and ELM


# ──────────────────────────────────────────────────────────────────────
# ELM hyperparameters
# ──────────────────────────────────────────────────────────────────────
ELM_HIDDEN     = 100        # hidden units
ELM_ACTIVATION = 'sigmoid'  # {'sigmoid', 'tanh', 'relu'}
ELM_C          = 0.1        # ridge regularization (0 = unregularized lstsq)


# ──────────────────────────────────────────────────────────────────────
# Slip mask thresholds (when to apply ELM correction)
# ──────────────────────────────────────────────────────────────────────
SLIP_THRESH = 0.025         # |slip angle| in rad
LONG_THRESH = 0.003         # |longitudinal slip| (dimensionless)
VX_THRESH   = 10.0 / 3.6    # m/s (10 km/h)


# ──────────────────────────────────────────────────────────────────────
# Training data files
# ──────────────────────────────────────────────────────────────────────
TRAIN_FILES = [
    os.path.join(TRAIN_DIR, f'Train_set_{i}.CSV') for i in range(1, 5)
]

# Files used to fit the ELM residual corrector (post-BiLSTM)
ELM_FIT_FILES = [
    os.path.join(ELM_TRAIN_DIR, f'Test_set_{tag}.CSV')
    for tag in ['L14_H', 'L16_H', 'L20_H']
]


# ──────────────────────────────────────────────────────────────────────
# Test sets and their rack-force ranges (for NRMSE normalization)
# ──────────────────────────────────────────────────────────────────────
TEST_FILES = {
    'gm_nor':              ('Test_set_gm_nor.CSV',              3283.46),
    'low_speed':           ('Test_set_low_speed.CSV',            368.49),
    'sine_15':             ('Test_set_sine_15.CSV',              239.28),
    'sine_60':             ('Test_set_sine_60.CSV',             1682.85),
    'stationary_steering': ('Test_set_stationary_steering.CSV',  228.98),
    'gm_agg':              ('Test_set_gm_agg.CSV',             3436.17),
    'gm_mu_0.4':           ('Test_set_gm_mu_0.4.CSV',          1592.37),
    'gm_mu_0.6':           ('Test_set_gm_mu_0.6.CSV',          2300.58),
}


# ──────────────────────────────────────────────────────────────────────
# Baseline weights filenames
# ──────────────────────────────────────────────────────────────────────
BIGRU_HIDDEN, BIGRU_LAYERS = 24, 1
LSTM_HIDDEN,  LSTM_LAYERS  = 96, 1

"""
Model definitions: BiLSTM regressor + ELM residual corrector.
"""

import torch
import torch.nn as nn


class BiLSTMRegressor(nn.Module):
    """
    Stacked bidirectional LSTM that maps a sequence of vehicle-dynamics
    features to a scalar rack-force value.

    Input  : (batch, seq_len, n_features)
    Output : (batch, 1)
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size,
                         device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size,
                         device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


class ELM:
    """
    Extreme Learning Machine with random hidden-layer weights and a closed-form
    output solver. Used to learn the BiLSTM residual on slip-dominant samples.

    The hidden weights (W, b) are drawn once at construction; only the output
    weights `beta` are fit via (regularized) least-squares.
    """

    def __init__(self, input_size, hidden_size, activation='sigmoid',
                 C=0.1, seed=42):
        torch.manual_seed(seed)
        self.W = torch.randn(input_size, hidden_size, dtype=torch.float64)
        self.b = torch.randn(hidden_size, dtype=torch.float64)
        self.beta = None
        self.activation = activation
        self.C = C
        self.input_size = input_size
        self.hidden_size = hidden_size

    def _H(self, X):
        Z = X.double() @ self.W + self.b
        if self.activation == 'sigmoid':
            return torch.sigmoid(Z)
        elif self.activation == 'tanh':
            return torch.tanh(Z)
        elif self.activation == 'relu':
            return torch.relu(Z)
        return torch.sigmoid(Z)

    def fit(self, X, y):
        """Solve (HᵀH + C·I) β = Hᵀ y when C>0, else lstsq(H, y)."""
        H = self._H(X)
        y = y.double()
        if self.C > 0:
            I = torch.eye(H.shape[1], dtype=torch.float64)
            self.beta = torch.linalg.solve(H.T @ H + self.C * I, H.T @ y)
        else:
            self.beta = torch.linalg.lstsq(H, y).solution

    def predict(self, X):
        if self.beta is None:
            raise RuntimeError("ELM has not been fit yet.")
        return (self._H(X) @ self.beta).float()

    def state_dict(self):
        return {
            'W': self.W, 'b': self.b, 'beta': self.beta,
            'activation': self.activation, 'C': self.C,
            'input_size': self.input_size, 'hidden_size': self.hidden_size,
        }

    @classmethod
    def from_state_dict(cls, sd):
        elm = cls(sd['input_size'], sd['hidden_size'],
                  activation=sd['activation'], C=sd['C'])
        elm.W = sd['W']; elm.b = sd['b']; elm.beta = sd['beta']
        return elm

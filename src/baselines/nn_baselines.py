"""
Baseline sequence models for comparison against BiLSTM + ELM.

  - BiGRU : bidirectional GRU + FC (matches the configuration in the paper)
  - LSTM  : uni-directional LSTM + FC (matches the configuration in the paper)
"""

import torch
import torch.nn as nn


class BiGRURegressor(nn.Module):
    """Bidirectional GRU regressor."""

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size,
                         device=x.device)
        out, _ = self.gru(x, h0)
        return self.fc(out[:, -1, :])


class LSTMRegressor(nn.Module):
    """Uni-directional stacked LSTM regressor."""

    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                         device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,
                         device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

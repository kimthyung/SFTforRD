import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Create necessary directories
def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + 'model/'):
        os.makedirs(path + 'model/')

def create_sequences(input_data, target_data, seq_length):
    x_seq, y_seq = [], []
    for i in range(len(input_data) - seq_length):
        x_seq.append(input_data[i:i + seq_length])
        y_seq.append(target_data[i + seq_length])
    return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

class LSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMmodel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
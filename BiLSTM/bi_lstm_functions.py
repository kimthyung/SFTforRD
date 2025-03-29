import os
import torch
import torch.nn as nn

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create necessary directories
def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + 'model/'):
        os.makedirs(path + 'model/')

def create_sequences(input_data, target_data, seq_length):
    x_seq, y_seq = [], []
    for i in range(len(input_data) - seq_length + 1):
        x_seq.append(input_data[i:i + seq_length])
        y_seq.append(target_data[i + seq_length - 1])
    return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

class BiLSTMmodel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(BiLSTMmodel, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Bidirectional, so hidden_size * 2

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    

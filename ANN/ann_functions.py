import torch
import torch.nn as nn
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create necessary directories
def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + 'model/'):
        os.makedirs(path + 'model/')

class ANNmodel(nn.Module):
    def __init__(self):
        super(ANNmodel, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
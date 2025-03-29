import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import silhouette_score
from lstm_functions import *

# Load the data
########### ######### ######## ######## ######### ########
########### ############ ############ ############ #######
########### ######### ######## ######## ######### ########
sequence_length = 50         ######### ################ ###
INCLUDE = 'YawRate'        ############ ############## ###
WHO = 'Human_v3'              ############ ############## ###
AI_MODEL = 'LSTM_org_loss'           ################ ########## ##
SENARIO = 'Training'          ########## ############ ######
num_epochs = 500           ########## ############ ######
########### ######### ######## ######## ######### ########
########### ############ ############ ############ #######
########### ######### ######## ######## ######### ########
DATA_NAME = f'{SENARIO}_{WHO}'
df = pd.read_csv(f'./DataSet/{DATA_NAME}.CSV')

# Calculate -7.03 * (TrqAlignFL + TrqAlignFR) and store in a new column
df['RackForce'] = (df['Car.TrqAlignFL'] + df['Car.TrqAlignFR']) / 0.5 # Tie rod length: 0.5m
# Multiply the specific column (e.g., 'TrqAlignFL') by 560
df['Steer.L.qp_modified'] = df['Steer.L.qp'] * 560

input_features = df[[f'Car.{INCLUDE}', 'Car.ay', 'Driver.Steer.Ang', 'Steer.L.q', 'Steer.L.qp_modified']]
output_feature = df['RackForce']

MODEL_PATH = f'sq_{sequence_length}'
MODE = f'include_{INCLUDE}'
PATH = f'{AI_MODEL}/{MODE}/{WHO}/{MODEL_PATH}/'

create_dirs(PATH)

hidden_size = 15
num_layers = 1
input_size = 5
output_size = 1

learning_rate = 0.001
# num_clusters = 6

# Select features and target for training
X_train = scaler_X.fit_transform(input_features.values)
y_train = scaler_y.fit_transform(output_feature.values.reshape(-1, 1))

# Seq data 
X_train, y_train = create_sequences(X_train, y_train, sequence_length)
x_seq, y_seq = X_train.to(device), y_train.to(device)

# DataLoader
batch_size = 128
train_dataset = TensorDataset(x_seq, y_seq)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = LSTMmodel(input_size=input_size,
                  hidden_size=hidden_size,
                  num_layers=num_layers,
                  output_size=output_size,
                  device=device).to(device)

# Loss function & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for seq, target in train_loader:
        # forward pass
        out = model(seq)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    loss_history.append(average_loss)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}')

print("Training finished!")

#Save scalers
joblib.dump(scaler_X, f'{PATH}model/scaler_X.pkl')
joblib.dump(scaler_y, f'{PATH}model/scaler_y.pkl')

#Save the model
torch.save(model, f'{PATH}model/model.pt')
torch.save(model.state_dict(), f'{PATH}model/model_state_dict.pt')
torch.save({
    'model' : model.state_dict(),
    'optimizer' : optimizer.state_dict()
}, PATH + 'model/' + 'all.tar')

#Plot loss history
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Results Saving
results_dir = PATH
FIGURE_NAME = f'Loss_history_{average_loss:.5f}.png'
plt.savefig(os.path.join(results_dir, FIGURE_NAME))

plt.show()

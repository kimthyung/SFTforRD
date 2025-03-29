# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset,random_split
# import numpy as np
# import pandas as pd
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import os
# import joblib
# from sklearn.metrics import silhouette_score
# from bi_lstm_functions import *
# from bi_lstm_data_create import * 
# from ELM_functions import TuningELM

# hidden_size = 30
# num_layers = 3
# input_size = 5
# output_size = 1
# num_epochs = 500
# learning_rate = 0.001

# # DataLoader
# batch_size = 128
# print(f'Shape of x_seq: {x_seq.shape}')
# print(f'Shape of y_seq: {y_seq.shape}')

# train_ratio = 0.6
# val_ratio = 0.2
# test_ratio = 0.2

# dataset = TensorDataset(x_seq, y_seq)
# train_size = int(train_ratio * len(dataset))
# val_size = int(val_ratio * len(dataset))
# test_size = len(dataset) - train_size - val_size

# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# # DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = BiLSTMmodel(input_size=input_size,
#                   hidden_size=hidden_size,
#                   num_layers=num_layers,
#                   output_size=output_size,
#                   device=device).to(device)

# # Loss function & Optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# loss_history = []
# val_loss_history = []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for seq, target in train_loader:
#         seq, target = seq.to(device), target.to(device)
#         # forward pass
#         out = model(seq)
#         loss = criterion(out, target)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     average_loss = running_loss / len(train_loader)
#     loss_history.append(average_loss)

#     # Validation phase
#     model.eval()
#     val_running_loss = 0.0
#     with torch.no_grad():
#         for val_seq, val_target in val_loader:
#             val_seq, val_target = val_seq.to(device), val_target.to(device)
#             val_out = model(val_seq)
#             val_loss = criterion(val_out, val_target)
#             val_running_loss += val_loss.item()

#     average_val_loss = val_running_loss / len(val_loader)
#     val_loss_history.append(average_val_loss)

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}, Validation Loss: {average_val_loss:.6f}')

# print("Training finished!")

# #Save the model
# torch.save(model, f'{PATH}model/model.pt')
# torch.save(model.state_dict(), f'{PATH}model/model_state_dict.pt')
# torch.save({
#     'model' : model.state_dict(),
#     'optimizer' : optimizer.state_dict()
# }, f'{PATH}model/all.tar')

# plt.plot(loss_history, label='Training Loss')
# plt.plot(val_loss_history, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# # Results Saving
# results_dir = PATH
# FIGURE_NAME = f'Loss_history_{average_loss:.5f}.png'
# plt.savefig(os.path.join(results_dir, FIGURE_NAME))

# plt.show()

# # Test the model
# model.eval()
# test_running_loss = 0.0
# with torch.no_grad():
#     for test_seq, test_target in test_loader:
#         test_seq, test_target = test_seq.to(device), test_target.to(device)
#         test_out = model(test_seq)
#         test_loss = criterion(test_out, test_target)
#         test_running_loss += test_loss.item()

# average_test_loss = test_running_loss / len(test_loader)
# print(f'Test Loss: {average_test_loss:.6f}')

######## APPLY EARLY STOPPING #############
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import silhouette_score
from bi_lstm_functions import *
from bi_lstm_data_create import * 
from ELM_functions import TuningELM

hidden_size = 30
num_layers = 3
input_size = 5
output_size = 1
num_epochs = 1000
learning_rate = 0.001

# Early Stopping parameters
patience = 500  # Number of epochs to wait before stopping if no improvement
min_delta = 0.001  # Minimum change in the monitored quantity to qualify as an improvement
best_val_loss = float('inf')
early_stop_counter = 0

# DataLoader
batch_size = 128
print(f'Shape of x_seq: {x_seq.shape}')
print(f'Shape of y_seq: {y_seq.shape}')

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

dataset = TensorDataset(x_seq, y_seq)
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = BiLSTMmodel(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    device=device).to(device)

# Loss function & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for seq, target in train_loader:
        seq, target = seq.to(device), target.to(device)
        # forward pass
        out = model(seq)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    loss_history.append(average_loss)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for val_seq, val_target in val_loader:
            val_seq, val_target = val_seq.to(device), val_target.to(device)
            val_out = model(val_seq)
            val_loss = criterion(val_out, val_target)
            val_running_loss += val_loss.item()

    average_val_loss = val_running_loss / len(val_loader)
    val_loss_history.append(average_val_loss)

    # Check for early stopping condition
    if average_val_loss < best_val_loss - min_delta:
        best_val_loss = average_val_loss
        early_stop_counter = 0  # Reset the counter if validation loss improves
        # Save the best model
        torch.save(model.state_dict(), f'{PATH}model/best_model_state_dict.pt')
    else:
        early_stop_counter += 1

    if early_stop_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.6f}, Validation Loss: {average_val_loss:.6f}')

print("Training finished!")

# Load the best model before testing
model.load_state_dict(torch.load(f'{PATH}model/best_model_state_dict.pt'))

# Plot the training and validation loss
plt.plot(loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Results Saving
results_dir = PATH
FIGURE_NAME = f'Loss_history_{average_loss:.5f}.png'
plt.savefig(os.path.join(results_dir, FIGURE_NAME))

plt.show()

# Test the model
model.eval()
test_running_loss = 0.0
with torch.no_grad():
    for test_seq, test_target in test_loader:
        test_seq, test_target = test_seq.to(device), test_target.to(device)
        test_out = model(test_seq)
        test_loss = criterion(test_out, test_target)
        test_running_loss += test_loss.item()

average_test_loss = test_running_loss / len(test_loader)
print(f'Test Loss: {average_test_loss:.6f}')

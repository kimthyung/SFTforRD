import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from ann_data_create import *
from ann_functions import *

num_epochs = 500

model = ANNmodel().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # back prop % optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_train_loss = running_loss / len(train_loader)
    loss_history.append(average_train_loss)
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_train_loss:.5f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_loss = 0.0
    y_preds = []
    y_tests = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        y_preds.append(outputs.cpu().numpy())
        y_tests.append(targets.cpu().numpy())

    average_test_loss = test_loss / len(test_loader)

    y_preds = np.concatenate(y_preds)
    y_tests = np.concatenate(y_tests)

mse = mean_squared_error(y_tests, y_preds)
rmse = np.sqrt(mse)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

# model save
torch.save(model, f'{PATH}model/model.pt')
torch.save(model.state_dict(), f'{PATH}model/model_state_dict.pt')
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict()
}, PATH + 'model/' + 'all.tar')

plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

results_dir = PATH 
FIGURE_NAME = f'Loss_history_{rmse:.4f}.png'
plt.savefig(os.path.join(results_dir, FIGURE_NAME))

plt.show()

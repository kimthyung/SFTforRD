import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from ann_functions import *
import joblib

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

VERSION = 'v1'
AI_MODEL = 'ANN'
MODEL_NAME = f'Benchmark_{AI_MODEL}'

now = datetime.now()
date = now.strftime("%m%d")
MODE = f'{date}_{VERSION}'
PATH = f'{AI_MODEL}/{MODE}/{MODEL_NAME}/'
create_dirs(PATH)

DATA_NUM = [1, 2, 3, 4]

# 1: General maneuver
# 2: Low speed
# 3: Stationary steering
# 4: U-turn

raw_features = []
raw_targets = []

for i in DATA_NUM:
    SENARIO = f'Train_set_{i}'
    DATA_NAME = f'{SENARIO}'

    df = pd.read_csv(f'./DataSet/{DATA_NAME}.CSV')

    df['RackForce'] = (df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2'])

    input_features = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
    output_feature = df['RackForce']

    raw_features.append(input_features)
    raw_targets.append(output_feature)

    print()
    print(f'<< Train set {i} >>')
    print('X_train: ', input_features.shape, 'y_train: ', output_feature.shape)

# Concatenate all features and targets
X = pd.concat(raw_features)
y = pd.concat(raw_targets)
df = pd.concat([X,y], axis=1)

X = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
y = df['RackForce']
#############################################################
# K-means
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters

balanced_df = df.groupby('cluster').apply(lambda x: x.sample(n=min(len(x), 50))).reset_index(drop=True)
X_balanced = balanced_df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
y_balanced = balanced_df['RackForce']
##############################################################

# StandardScaler
X_scaled = scaler_X.fit_transform(X_balanced)
y_scaled = scaler_y.fit_transform(y_balanced.values.reshape(-1, 1)).flatten()

print()
print(f'Number of samples in X_scaled: {X_scaled.shape}')
print(f'Number of samples in y_scaled: {y_scaled.shape}')

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

batch_size = 64
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train).view(-1, 1)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test).view(-1, 1)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Save scalers
joblib.dump(scaler_X, f'{PATH}model/scaler_X.pkl')
joblib.dump(scaler_y, f'{PATH}model/scaler_y.pkl')

print()
print(f"Train DataLoader: {len(train_loader)} batches")
print(f"Test DataLoader: {len(test_loader)} batches")
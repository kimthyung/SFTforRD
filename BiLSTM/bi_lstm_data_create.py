import os
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pandas as pd
from bi_lstm_functions import create_dirs, create_sequences
from datetime import datetime
import joblib

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()

x_seq = []
y_seq = []

VERSION = 'v1'
SEQUENCE_LENGTH = 10
AI_MODEL = 'BiLSTM'

MODEL_PATH = f'sq_{SEQUENCE_LENGTH}'
now = datetime.now()
date = now.strftime("%m%d")
MODE = f'{date}_{VERSION}'
PATH = f'{AI_MODEL}/{MODE}/{MODEL_PATH}/'
create_dirs(PATH)

DATA_NUM = [1, 2, 3, 4]

# 1: General maneuver
# 2: Low speed
# 3: Stationary steering
# 4: U-turn

# Placeholder for raw features and targets
raw_features = []
raw_targets = []

for i in DATA_NUM:
    SENARIO = f'Train_set_{i}'
    DATA_NAME = f'{SENARIO}'

    df = pd.read_csv(f'./DataSet/{DATA_NAME}.CSV')

    df['RackForce'] = (df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2'])

    # input_features = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Steer.L.q', 'Steer.L.qp_modified']]
    input_features = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
    output_feature = df['RackForce']

    raw_features.append(input_features)
    raw_targets.append(output_feature)

    print()
    print(f'<< Train set {i} >>')
    print('X_train: ',input_features.shape,'y_train: ',output_feature.shape)

# Concatenate all features and targets
raw_features = pd.concat(raw_features)
raw_targets = pd.concat(raw_targets)

# Fit and transform the scalers
X_train = scaler_X.fit_transform(raw_features.values)
y_train = scaler_y.fit_transform(raw_targets.values.reshape(-1, 1))

# Save scalers
joblib.dump(scaler_X, f'{PATH}model/scaler_X.pkl')
joblib.dump(scaler_y, f'{PATH}model/scaler_y.pkl')

# Create sequences
X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)

x_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
y_seq = torch.tensor(y_train_seq, dtype=torch.float32).to(device)

print(f'Total number of sequences: {x_seq.size(0)}')
print(f'Shape of x_seq: {x_seq.shape}')
print(f'Shape of y_seq: {y_seq.shape}')

# import os
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
# import pandas as pd
# from bi_lstm_functions import create_dirs, create_sequences
# from datetime import datetime
# import joblib

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Feature scaling
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()

# VERSION = 'v8'
# SEQUENCE_LENGTH = 10
# AI_MODEL = 'BiLSTM'

# MODEL_PATH = f'sq_{SEQUENCE_LENGTH}'
# now = datetime.now()
# date = now.strftime("%m%d")
# MODE = f'{date}_{VERSION}'
# PATH = f'{AI_MODEL}/{MODE}/{MODEL_PATH}/'
# create_dirs(PATH)

# DATA_NUM = [1, 2, 3, 4]

# # 1: General maneuver
# # 2: Low speed
# # 3: Stationary steering
# # 4: U-turn

# # Placeholder for raw features and targets
# raw_features = []
# raw_targets = []

# for i in DATA_NUM:
#     SCENARIO = f'Train_set_{i}'
#     DATA_NAME = f'{SCENARIO}'

#     df = pd.read_csv(f'./DataSet/{DATA_NAME}.CSV')

#     df['RackForce'] = (df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2'])

#     input_features = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
#     output_feature = df['RackForce']

#     raw_features.append(input_features)
#     raw_targets.append(output_feature)

# # Concatenate all features and targets
# raw_features = pd.concat(raw_features)
# raw_targets = pd.concat(raw_targets)

# # Fit and transform the scalers
# X_train = scaler_X.fit_transform(raw_features.values)
# y_train = scaler_y.fit_transform(raw_targets.values.reshape(-1, 1))

# # Save scalers
# joblib.dump(scaler_X, f'{PATH}model/scaler_X.pkl')
# joblib.dump(scaler_y, f'{PATH}model/scaler_y.pkl')

# # Create sequences
# X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)

# x_seq = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
# y_seq = torch.tensor(y_train_seq, dtype=torch.float32).to(device)

# # Flatten the sequences for clustering
# X_flattened = x_seq.view(x_seq.shape[0], -1).cpu().numpy()
# ####################################################################
# def find_optimal_clusters(data, max_k):
#     silhouette_scores = []
#     inertias = []

#     K_range = range(2, max_k + 1)

#     for k in K_range:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(data)
#         inertias.append(kmeans.inertia_)
#         score = silhouette_score(data, kmeans.labels_)
#         silhouette_scores.append(score)

#     # Plot Elbow Method
#     plt.figure(figsize=(10, 5))
#     plt.plot(K_range, inertias, marker='o')
#     plt.title('Elbow Method for Sequence Data')
#     plt.xlabel('Number of clusters (K)')
#     plt.ylabel('Inertia')
#     plt.grid(True)
#     plt.show()

#     # Plot Silhouette Scores
#     plt.figure(figsize=(10, 5))
#     plt.plot(K_range, silhouette_scores, marker='o')
#     plt.title('Silhouette Scores for Sequence Data')
#     plt.xlabel('Number of clusters (K)')
#     plt.ylabel('Silhouette Score')
#     plt.grid(True)
#     plt.show()

#     # Return the optimal number of clusters based on the highest silhouette score
#     optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
#     return optimal_k

# # Determine the optimal number of clusters
# # max_clusters = 15  # You can adjust this number based on your dataset
# # optimal_k = find_optimal_clusters(X_flattened, max_clusters)
# # print(f'Optimal number of clusters based on silhouette score: {optimal_k}')

# ################################################################
# # Set the number of clusters to 7 and apply K-means clustering
# n_clusters = 20
# kmeans = KMeans(n_clusters=n_clusters, random_state=42)
# clusters = kmeans.fit_predict(X_flattened)

# # Print the number of data points in each cluster
# cluster_counts = pd.Series(clusters).value_counts()
# print(f'\nNumber of data points per cluster:\n{cluster_counts}')

# # Optionally, you can now segment x_seq and y_seq according to clusters for further analysis or training
# clustered_x_seq = {}
# clustered_y_seq = {}

# for k in range(n_clusters):
#     indices = [i for i, cluster_id in enumerate(clusters) if cluster_id == k]
#     clustered_x_seq[k] = x_seq[indices]
#     clustered_y_seq[k] = y_seq[indices]

# # Re-combine clustered sequences into a single tensor for each type
# combined_x_seq = torch.cat([clustered_x_seq[k] for k in clustered_x_seq.keys()], dim=0)
# combined_y_seq = torch.cat([clustered_y_seq[k] for k in clustered_y_seq.keys()], dim=0)

# # Re-assign combined tensors back to x_seq and y_seq
# x_seq = combined_x_seq
# y_seq = combined_y_seq

# # Printing shapes to verify
# print(f'Combined x_seq shape: {x_seq.shape}')
# print(f'Combined y_seq shape: {y_seq.shape}')

# import pandas as pd
# import torch
# from sklearn.preprocessing import StandardScaler
# import joblib
# import numpy as np
# from ELM_functions import *
# from bi_lstm_functions import *

# # Bi-LSTM parameter
# AI_MODEL = 'BiLSTM'
# sequence_length = 10
# hidden_size = 30
# num_layers = 3
# input_size = 5
# output_size = 1
# date = '0809'
# MODEL_VERSION = 'v2'


# MODEL_PATH = f'sq_{sequence_length}'
# MODE = f'{date}_{MODEL_VERSION}'
# PATH = f'{AI_MODEL}/{MODE}/{MODEL_PATH}/'

# # Model & Scaler Load
# model = BiLSTMmodel(input_size=input_size,
#                     hidden_size=hidden_size,
#                     num_layers=num_layers,
#                     output_size=output_size,
#                     device=device).to(device)

# ##################################

# scaler_ELM = StandardScaler()
# SCENARIO_NUMS = ['L14_H', 'L16_H', 'L20_H']  
# ELM_MODEL_VERSION = 'v2'
# # slip_angle_threshold = 0.01
# # long_slip_threshold = 0.003

# slip_angle_threshold = 0.01
# long_slip_threshold = 0.003

# X_train_list = []
# y_train_list = []

# for SCENARIO_NUM in SCENARIO_NUMS:
#     SCENARIO = f'Test_{SCENARIO_NUM} (v-Friction)'

#     df = pd.read_csv(f'./DataSet/Test_set_{SCENARIO_NUM}.CSV')

#     ELM_inputs_df = df[['Car.SlipAngleFL', 'Car.SlipAngleFR', 'Car.YawRate', 'Driver.Steer.Ang', 'Car.ay']]
    
#     df['RackForce'] = (df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2'])

#     input_features = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
#     output_feature = df['RackForce']

#     scaler_X = joblib.load(f'{PATH}model/scaler_X.pkl')
#     scaler_y = joblib.load(f'{PATH}model/scaler_y.pkl')

#     X_test = scaler_X.transform(input_features.values)
#     y_test = scaler_y.transform(output_feature.values.reshape(-1, 1))

#     X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
#     X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
#     y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

#     model.load_state_dict(torch.load(f'{PATH}model/model_state_dict.pt'))

#     model.eval()

#     with torch.no_grad():
#         predictions = model(X_test_tensor)

#     y_pred = scaler_y.inverse_transform(predictions.cpu().numpy())
#     y_real = scaler_y.inverse_transform(y_test_seq.cpu().numpy())

#     rack_force_actuals = torch.tensor(y_real, dtype=torch.float32)
#     rack_force_predictions = torch.tensor(y_pred, dtype=torch.float32)
#     errors = rack_force_actuals - rack_force_predictions

#     friction_values = (df[['Car.muRoadFL']].values[sequence_length-1:] + df[['Car.muRoadFR']].values[sequence_length-1:]) / 2
#     slip_angle_values = (df[['Car.SlipAngleFL']].values[sequence_length-1:] + df[['Car.SlipAngleFR']].values[sequence_length-1:]) / 2
#     long_slip_values = (df[['Car.LongSlipFL']].values[sequence_length-1:] + df[['Car.LongSlipFR']].values[sequence_length-1:]) / 2
#     car_vx_values = df[['Car.vx']].values[sequence_length-1:]

#     condition_mask = (car_vx_values >= 10/3.6).squeeze() & \
#         ((abs(slip_angle_values).squeeze() >= slip_angle_threshold) | (abs(long_slip_values).squeeze() >= long_slip_threshold))

#     X_train_list.append(ELM_inputs_df.values[sequence_length-1:][condition_mask])
#     y_train_list.append(errors[condition_mask])

# X_train = np.concatenate(X_train_list, axis=0)
# y_train = torch.cat(y_train_list, dim=0)

# # perm = torch.randperm(X_train.shape[0])

# # X_train = X_train[perm.numpy()]
# # y_train = y_train[perm]

# # X_train_scaled = scaler_ELM.fit_transform(X_train)

# # X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
# # y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
# ############################################

# # Combine X_train and y_train for clustering
# # Ensure y_train is flattened to 2D by squeezing unnecessary dimensions
# y_train = y_train.squeeze()  # This removes any extra dimensions, ensuring it's 2D like X_train

# # Combine X_train and y_train for clustering
# train_combined = np.hstack((X_train, y_train.unsqueeze(1).cpu().numpy()))

# ################# Elbow n Silhouette ############
# # plot_elbow_method(train_combined, max_clusters=20)
# # plot_silhouette_scores(train_combined, max_clusters=20)
# #######################################################

# # Apply K-means clustering
# kmeans = KMeans(n_clusters=4, random_state=42)
# clusters = kmeans.fit_predict(train_combined)

# # Add the cluster labels
# train_combined_with_cluster = np.hstack((train_combined, clusters.reshape(-1, 1)))

# # Balance the data by selecting a fixed number of samples from each cluster
# balanced_data = pd.DataFrame(train_combined_with_cluster).groupby(train_combined_with_cluster[:, -1]).apply(lambda x: x.sample(n=min(len(x), 50))).reset_index(drop=True).values

# # Split back into X_train and y_train
# X_train_balanced = balanced_data[:, :-2]  # All columns except the last two (y_train and cluster)
# y_train_balanced = balanced_data[:, -2]  # Second last column for y_train

# # Shuffle the balanced data
# perm = torch.randperm(X_train_balanced.shape[0])
# X_train_balanced = X_train_balanced[perm.numpy()]
# y_train_balanced = y_train_balanced[perm.numpy()]

# # StandardScaler
# X_train_scaled = scaler_ELM.fit_transform(X_train_balanced)

# X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.float32).to(device)

# ELM_PATH = f'/home/jwon/ros2_ws/src/LSTMmodel/BiLSTM/0809_v2/sq_10/ELM/{ELM_MODEL_VERSION}/'
# create_dirs(ELM_PATH)

# joblib.dump(scaler_ELM, f'{ELM_PATH}/scaler_ELM.pkl')
# print()
# print()
# print("Data preparation complete.")
###################################
########### Original Version ###########

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from ELM_functions import *
from BiLSTM.bi_lstm_functions import *
from data_processing import *

# Bi-LSTM parameter
AI_MODEL = 'BiLSTM'
sequence_length = 10
hidden_size = 30
num_layers = 3
input_size = 5
output_size = 1
date = '0809'
MODEL_VERSION = 'v2'


MODEL_PATH = f'sq_{sequence_length}'
MODE = f'{date}_{MODEL_VERSION}'
PATH = f'{AI_MODEL}/{MODE}/{MODEL_PATH}/'

# Model & Scaler Load
model = BiLSTMmodel(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=output_size,
                    device=device).to(device)

##################################

scaler_ELM = StandardScaler()
SCENARIO_NUMS = ['L14_H', 'L16_H', 'L20_H']  
ELM_MODEL_VERSION = 'v3'
# slip_angle_threshold = 0.01
# long_slip_threshold = 0.003

slip_angle_threshold = 0.01
long_slip_threshold = 0.003

X_train_list = []
y_train_list = []

for SCENARIO_NUM in SCENARIO_NUMS:
    SCENARIO = f'Test_{SCENARIO_NUM} (v-Friction)'

    df = pd.read_csv(f'./DataSet/Test_set_{SCENARIO_NUM}.CSV')
    df = data_processing(df)
    
    ELM_inputs_df = df[['Car.SlipAngleFL', 'Car.SlipAngleFR', 'Car.YawRate', 'Driver.Steer.Ang', 'Car.ay']]
    
    df['RackForce'] = (df['Car.CFL.GenFrc2'] + df['Car.CFR.GenFrc2'])

    input_features = df[['Car.YawRate', 'Car.ay', 'Driver.Steer.Ang', 'Driver.Steer.AngVel', 'Car.vx']]
    output_feature = df['RackForce']

    scaler_X = joblib.load(f'{PATH}model/scaler_X.pkl')
    scaler_y = joblib.load(f'{PATH}model/scaler_y.pkl')

    X_test = scaler_X.transform(input_features.values)
    y_test = scaler_y.transform(output_feature.values.reshape(-1, 1))

    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).to(device)

    model.load_state_dict(torch.load(f'{PATH}model/model_state_dict.pt'))

    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)

    y_pred = scaler_y.inverse_transform(predictions.cpu().numpy())
    y_real = scaler_y.inverse_transform(y_test_seq.cpu().numpy())

    rack_force_actuals = torch.tensor(y_real, dtype=torch.float32)
    rack_force_predictions = torch.tensor(y_pred, dtype=torch.float32)
    errors = rack_force_actuals - rack_force_predictions

    friction_values = (df[['Car.muRoadFL']].values[sequence_length-1:] + df[['Car.muRoadFR']].values[sequence_length-1:]) / 2
    slip_angle_values = (df[['Car.SlipAngleFL']].values[sequence_length-1:] + df[['Car.SlipAngleFR']].values[sequence_length-1:]) / 2
    long_slip_values = (df[['Car.LongSlipFL']].values[sequence_length-1:] + df[['Car.LongSlipFR']].values[sequence_length-1:]) / 2
    car_vx_values = df[['Car.vx']].values[sequence_length-1:]

    condition_mask = (car_vx_values >= 10/3.6).squeeze() & \
        ((abs(slip_angle_values).squeeze() >= slip_angle_threshold) | (abs(long_slip_values).squeeze() >= long_slip_threshold))

    X_train_list.append(ELM_inputs_df.values[sequence_length-1:][condition_mask])
    y_train_list.append(errors[condition_mask])

X_train = np.concatenate(X_train_list, axis=0)
y_train = torch.cat(y_train_list, dim=0)

perm = torch.randperm(X_train.shape[0])

X_train = X_train[perm.numpy()]
y_train = y_train[perm]

X_train_scaled = scaler_ELM.fit_transform(X_train)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

ELM_PATH = f'/home/jwon/ros2_ws/src/LSTMmodel/BiLSTM/0809_v2/sq_10/ELM/{ELM_MODEL_VERSION}/'
create_dirs(ELM_PATH)

joblib.dump(scaler_ELM, f'{ELM_PATH}/scaler_ELM.pkl')
print()
print()
print("Data preparation complete.")

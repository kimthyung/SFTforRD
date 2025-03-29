import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from params import *
from functions import *

MODE = '3'
SENARIO = f'Train_set_GP_{MODE}'

DATA_NAME = f'{SENARIO}'
df = pd.read_csv(f'./DataSet/Train_set_GP_3.CSV')


scaler = StandardScaler()
MODEL = 4

yawrate      = df['Car.YawRate']
ay           = df['Car.ay']
v            = df['Car.v']
ax           = df['Car.ax']
delta        = (df['Car.SteerAngleFL']+df['Car.SteerAngleFR'])/2
delta_dot    = np.diff(delta)/dt
delta_dot    = np.append(delta_dot, delta_dot[-1]) 
RackForce_GT = (df['Car.CFL.GenFrc2']+df['Car.CFR.GenFrc2'])

delta = delta.to_numpy()
v = v.to_numpy()
ay = ay.to_numpy()
yawrate = yawrate.to_numpy()
ax = ax.to_numpy()

F_move_values = []
F_stat_values = []

v = np.where(v <= v_threshold, v_threshold, v)

for i in range(len(df)):

    u = np.array([delta[i], v[i]])
    z = np.array([ay[i], yawrate[i]])

    alpha_f, alpha_r = slip_angle(x, u)
    F_zf, F_zr = tire_normal_force(ax[i])
    F_fy_stat = lateral_force_ode(alpha_f, F_zf)
    F_ry_stat = lateral_force_ode(alpha_r, F_zr)

    x_pred, P_pred = ekf_predict(x, P, u, F_fy_stat, F_ry_stat)
    x, P = ekf_update(x_pred, P_pred, z, u)

    # Rack Force Estimation
    F_move = high_speed_rack_force(x)
    F_stat, theta = stat_rack_force(theta, delta_dot[i])

    F_move_values.append(F_move)
    F_stat_values.append(F_stat)

if np.isnan(F_move_values).any() or np.isinf(F_move_values).any():
    print("Warning: F_move_values contains NaN or inf values.")

    nan_inf_indices = np.where(np.isnan(F_move_values) | np.isinf(F_move_values))[0]
    print(f"NaN or inf values found at indices: {nan_inf_indices}")
else:
    print("F_move_values contains no NaN or inf values.")

if np.isnan(F_stat_values).any() or np.isinf(F_stat_values).any():
    print("Warning: F_stat_values contains NaN or inf values.")

    nan_inf_indices = np.where(np.isnan(F_stat_values) | np.isinf(F_stat_values))[0]
    print(f"NaN or inf values found at indices: {nan_inf_indices}")
else:
    print("F_stat_values contains no NaN or inf values.")

processed_df = pd.DataFrame({
    'F_move': F_move_values,
    'F_stat': F_stat_values,
    'v': v,
    'RackForce_GT': RackForce_GT
})

stationary = processed_df[processed_df['v'] < 10/3.6]
low_speed = processed_df[(processed_df['v'] >= 10/3.6) & (processed_df['v'] < 30/3.6)]
high_speed = processed_df[(processed_df['v'] > 30/3.6)]

stationary = stationary.sample(frac=0.05)
low_speed_sample = low_speed.sample(frac=0.09)
high_speed_sample = high_speed.sample(frac=0.005)

sampled_df = pd.concat([stationary, low_speed_sample, high_speed_sample])

print("stationary_length : ", len(stationary))
print("low_speed_length : ", len(low_speed_sample))
print("high_speed_length : ", len(high_speed_sample))

plt.figure(figsize=(10, 6))
plt.plot(RackForce_GT, label='Actual Rack Force', linewidth=3.5, color='black')
plt.plot(F_move_values, label='F_move', linewidth=2, color='red')
plt.plot(F_stat_values, label='F_stat', linewidth=2, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Rack Force (N)')
plt.title('Comparison of Actual and F_pred')
plt.legend()
# plt.show()

X = sampled_df[['F_move', 'F_stat', 'v']]
Y = sampled_df['RackForce_GT']

X = scaler.fit_transform(X)

joblib.dump(scaler, f'model_based/model_{MODEL}/scaler.pkl')

kernel = C(1.0, (1e-3, 1e7)) * Matern(length_scale=2.5, nu=1.5) + WhiteKernel(noise_level=1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
gp.fit(X, Y)

print("\nTrain finished.\n")

Y_pred = gp.predict(X)
mse_train = mean_squared_error(Y, Y_pred)
print(f"MSE: {mse_train:.4f}")

r2_train = r2_score(Y, Y_pred)
print(f"R²: {r2_train:.4f}")

scores = cross_val_score(gp, X, Y, cv=5, scoring='neg_mean_squared_error')
mse_cv = -np.mean(scores)
print(f"MSE: {mse_cv:.4f}")

joblib.dump(gp, f'model_based/model_{MODEL}/gpr_model.pkl')
print("GP model saved.")

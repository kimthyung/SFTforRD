import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from params import *
from functions import *
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data_processing import *

# MODE = '3'
# SENARIO = f'Train_set_GP_{MODE}'
MODE = 'sine_60'
SENARIO = f'Test_set_{MODE}'
DATA_NAME = f'{SENARIO}'
df = pd.read_csv(f'./DataSet/{DATA_NAME}.CSV')
df = data_processing(df)

yawrate = df['Car.YawRate']
ay = df['Car.ay']
v = df['Car.v']
ax = df['Car.ax']
delta = (df['Car.SteerAngleFL']+df['Car.SteerAngleFR'])/2
RackForce_GT = (df['Car.CFL.GenFrc2']+df['Car.CFR.GenFrc2'])

# x[0] : beta
# x[1] : yawrate
# x[2] : F_fy
# x[3] : F_ry

# u[0] : delta
# u[1] : v

calculated_rack_forces = []
beta_vals, yaw_rate_vals, F_fy_vals, F_ry_vals, v_vals = [], [], [], [], []
alpha_f_vals, alpha_r_vals = [], []
F_zf_vals, F_zr_vals = [], []

v = np.where(v <= v_threshold, v_threshold, v)

for i in range(len(df)):

    u = np.array([delta[i], v[i]])
    z = np.array([ay[i], yawrate[i]])

    alpha_f, alpha_r = slip_angle(x, u)

    F_zf, F_zr = tire_normal_force(ax[i])

    F_fy_stat = lateral_force_ode(alpha_f, F_zf)
    F_ry_stat = lateral_force_ode(alpha_r, F_zr)

    # EKF Predict
    x_pred, P_pred = ekf_predict(x, P, u, F_fy_stat, F_ry_stat)

    # EKF Update
    x, P = ekf_update(x_pred, P_pred, z, u)

    # Rack Force Estimation
    rack_force = high_speed_rack_force(x)
    
    # 계산된 rack_force를 리스트에 추가
    calculated_rack_forces.append(rack_force)

    beta_vals.append(x[0])
    yaw_rate_vals.append(x[1])
    F_fy_vals.append(x[2])
    F_ry_vals.append(x[3])

    v_vals.append(u[1])
    alpha_f_vals.append(alpha_f)
    alpha_r_vals.append(alpha_r)
    F_zf_vals.append(F_zf)
    F_zr_vals.append(F_zr)

    # print(f"Iteration {i + 1}: Beta = {x[0]:.4f}, YawRate = {x[1]:.2f}, Rack Force = {rack_force:.2f} N")

# plt.figure(figsize=(10, 6))
# plt.plot(RackForce_GT[200:len(calculated_rack_forces)-200], label='Actual Rack Force', linewidth=3.5, color='black')
# plt.plot(calculated_rack_forces[200:-200], label='Calculated Rack Force', linewidth=2, color='red')
# plt.xlabel('Iteration')
# plt.ylabel('Rack Force (N)')
# plt.title('Comparison of Actual and Calculated Rack Force')
# plt.legend()
# plt.show()

valid_start = 200
valid_end = len(calculated_rack_forces) - 200

plt.figure(figsize=(10, 6))
plt.plot(range(valid_start, valid_end), RackForce_GT[valid_start:valid_end], label='Actual Rack Force', linewidth=3.5, color='black')
plt.plot(range(valid_start, valid_end), calculated_rack_forces[valid_start:valid_end], label='Calculated Rack Force', linewidth=2, color='red')
plt.xlabel('Iteration')
plt.ylabel('Rack Force (N)')
plt.title('Comparison of Actual and Calculated Rack Force')
plt.legend()
plt.show()

if np.isnan(calculated_rack_forces).any() or np.isinf(calculated_rack_forces).any():
    print("Warning: F_move_values contains NaN or inf values.")

    nan_inf_indices = np.where(np.isnan(calculated_rack_forces) | np.isinf(calculated_rack_forces))[0]
    print(f"NaN or inf values found at indices: {nan_inf_indices}")
else:
    print("F_move_values contains no NaN or inf values.")

y_pred = calculated_rack_forces
y_real = RackForce_GT[:len(calculated_rack_forces)]

rmse_model_based = np.sqrt(np.mean((y_pred - y_real) ** 2))
nrmse_model_based = rmse_model_based / (np.max(y_real) - np.min(y_real))

print(f"NRMSE (model_based): {nrmse_model_based:.4f}")

fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)

axs[0].plot(alpha_f_vals, label='alpha front (alpha_f)')
axs[0].plot((df['Car.SlipAngleFR']+df['Car.SlipAngleFL'])/2, label='alpha front Real')
axs[0].set_ylabel('alphaf (rad)')
axs[0].legend()

axs[1].plot(F_zf_vals, label='Front Tire Normal Force [N]')
axs[1].plot((df['Car.CFR.Tire.FrcC.z']+df['Car.CFL.Tire.FrcC.z']), label='GT [N]')
axs[1].set_ylabel('F_zf (N)')
axs[1].legend()

axs[2].plot(F_zr_vals, label='Rear Tire Normal Force [N]')
axs[2].plot((df['Car.CRR.Tire.FrcC.z']+df['Car.CRL.Tire.FrcC.z']), label='GT [N]')
axs[2].set_ylabel('F_zr (N)')
axs[2].legend()

axs[3].plot(F_fy_vals, label='Front Lateral Force (F_fy)')
axs[3].plot((df['Car.CFL.Tire.FrcC.y']+df['Car.CFR.Tire.FrcC.y']), label='F_fy GT')
axs[3].set_ylabel('F_fy (N)')
axs[3].legend()

axs[4].plot(F_ry_vals, label='Rear Lateral Force (F_ry)')
axs[4].plot((df['Car.CRL.Tire.FrcC.y']+df['Car.CRR.Tire.FrcC.y']), label='F_ry GT')
axs[4].set_ylabel('F_ry (N)')
axs[4].legend()

axs[5].plot(beta_vals, label='Beta (β)')
axs[5].plot(df['Car.SideSlipAngle'], label='Beta (β) GT')
axs[5].set_ylabel('Beta (rad)')
axs[5].legend()

axs[6].plot(yaw_rate_vals, label='Yaw Rate')
axs[6].plot(yawrate, label='Yaw Rate_real')
axs[6].set_ylabel('Yaw Rate (rad/s)')
axs[6].legend()

fig, axs = plt.subplots(7, 1, figsize=(10, 12), sharex=True)

axs[0].plot(F_fy_vals, label='Front Lateral Force (F_fy)')
axs[0].set_ylabel('F_fy (N)')
axs[0].legend()

axs[1].plot((df['Car.CFL.Tire.FrcC.y']+df['Car.CFR.Tire.FrcC.y']), label='F_fy GT',color='red')
axs[1].set_ylabel('F_fy (N)')
axs[1].legend()

axs[2].plot(F_ry_vals, label='Rear Lateral Force (F_ry)')
axs[2].set_ylabel('F_ry (N)')
axs[2].legend()

axs[3].plot((df['Car.CRL.Tire.FrcC.y']+df['Car.CRR.Tire.FrcC.y']), label='F_ry GT',color='red')
axs[3].set_ylabel('F_ry (N)')
axs[3].legend()

axs[4].plot(alpha_f_vals, label='alpha front (alpha_f)')
axs[4].set_ylabel('alphaf (rad)')
axs[4].legend()

axs[5].plot((df['Car.SlipAngleFR']+df['Car.SlipAngleFL'])/2, label='GT',color='red')
axs[5].set_ylabel('alphaf (rad)')
axs[5].legend()

axs[6].plot(df['Car.v']*3.6, label='Car.v')
axs[6].set_ylabel('velocity (kph)')
axs[6].legend()

plt.xlabel('Iteration')
plt.suptitle('State Variables over Time')
plt.show()

# rack_force_df = pd.DataFrame({
#     'Calculated Rack Force': calculated_rack_forces
# })
 
# rack_force_df.to_csv('model_based\predictions_csv\model_based_H111_H.csv', index=False)
# print("Rack Force 데이터가 'model_based_H111_H.csv'로 저장되었습니다.")
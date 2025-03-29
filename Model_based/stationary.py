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

# MODE = '2'
# SENARIO = f'Train_set_GP_{MODE}'

MODE = 'stationary_steering'
SENARIO = f'Test_set_{MODE}'
DATA_NAME = f'{SENARIO}'
df = pd.read_csv(f'./DataSet/{DATA_NAME}.CSV')

delta = (df['Car.SteerAngleFL']+df['Car.SteerAngleFR'])/2
delta_dot = np.diff(delta)/dt
delta_dot = np.append(delta_dot, delta_dot[-1]) 
RackForce_GT = (df['Car.CFL.GenFrc2']+df['Car.CFR.GenFrc2'])

# x[0] : beta
# x[1] : yawrate
# x[2] : F_fy
# x[3] : F_ry

# u[0] : delta
# u[1] : v

calculated_rack_forces = []
theta_values = []
theta_dot_values = []

for i in range(len(df)):

    rack_force, theta = stat_rack_force(theta, delta_dot[i])

    calculated_rack_forces.append(rack_force) 
    theta_values.append(theta)
    theta_dot_values.append(theta_dot)
    # print(f"Iteration {i + 1}:Rack Force = {rack_force:.2f} N")

plt.figure(figsize=(10, 6))
plt.plot(delta*14.25*180/np.pi, calculated_rack_forces, label='Calculated Rack Force', linewidth=2, color='red')
plt.plot(delta*14.25*180/np.pi, RackForce_GT[:len(calculated_rack_forces)], label='Actual Rack Force', linewidth=3.5, color='black')
plt.xlabel('Steering wheel angle (deg)')
plt.ylabel('Rack Force (N)')
plt.title('Comparison of Actual and Calculated Rack Force')
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(RackForce_GT[:len(calculated_rack_forces)], label='Actual Rack Force', linewidth=3.5, color='black')
plt.plot(calculated_rack_forces, label='Calculated Rack Force', linewidth=2, color='red')
plt.xlabel('Steering wheel angle (deg)')
plt.ylabel('Rack Force (N)')
plt.title('Comparison of Actual and Calculated Rack Force')
plt.legend()

if np.isnan(calculated_rack_forces).any() or np.isinf(calculated_rack_forces).any():
    print("Warning: F_stat_values contains NaN or inf values.")
    nan_inf_indices = np.where(np.isnan(calculated_rack_forces) | np.isinf(calculated_rack_forces))[0]
    print(f"NaN or inf values found at indices: {nan_inf_indices}")
else:
    print("F_stat_values contains no NaN or inf values.")

# plt.figure()
# plt.plot(theta_values, label='theta_values')
# plt.legend()

# plt.figure()
# plt.plot(delta,label='delta_dot')
# plt.legend()

# plt.figure()
# plt.plot(theta_dot_values,label='theta_dot')
# plt.legend()

plt.show()


# Rack Force 데이터를 DataFrame으로 저장
rack_force_df = pd.DataFrame({
    'Calculated Rack Force': calculated_rack_forces
})


# rack_force_df.to_csv('model_based\predictions_csv\model_based_H111_H_stationary.csv', index=False)

# print("Rack Force 데이터가 'model_based_H111_H.csv'로 저장되었습니다.")
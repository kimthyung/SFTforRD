import numpy as np
from params import *

def slip_angle(x, u):
    
    # alpha_f = u[0] - np.arctan((u[1]*np.sin(x[0])+l_f*x[1])/(u[1]*np.cos(x[0])))
    # alpha_r = -np.arctan((u[1]*np.sin(x[0])-l_r*x[1])/(u[1]*np.cos(x[0])))
    # if u[1] <= 2:
    #     alpha_f = alpha_f/1000
    #     alpha_r = alpha_r/1000

    alpha_f = u[0] - np.arctan((u[1]*np.sin(x[0])+l_f*x[1])/(u[1]*np.cos(x[0])))
    alpha_r = -np.arctan((u[1]*np.sin(x[0])-l_r*x[1])/(u[1]*np.cos(x[0])))

    return alpha_f, alpha_r

def tire_normal_force(ax):
    F_zf = m * ((-ax*h_cg + g*l_r )/(l_f+l_r))
    F_zr = m * ((ax*h_cg + g*l_f )/(l_f+l_r))
    return F_zf, F_zr

def lateral_force_ode(alpha, F_z):
    F_y_stat = mu * F_z * (1 - np.exp(-K * np.abs(alpha))) * np.sign(alpha)
    return F_y_stat

def f(x, u, F_fy_stat, F_ry_stat):

    beta_dot = -x[1] + (x[2]*np.cos(u[0]-x[0])/(m*u[1])) + (x[3]*np.cos(x[0])/(m*u[1]))
    # -x[1] + (1/(m*u[1]))*(x[2] * np.cos(u[0] - x[0]) + x[3] * np.cos(x[0]))
    # -x[1] + (x[2]*np.cos(u[0] - x[0])/(m*u[1])) + (x[3]*np.cos(x[0])/(m*u[1]))
    yaw_rate_dot = (1/I_zz)*(x[2]*l_f*np.cos(u[0]) - x[3] * l_r)
    F_fy_dot = -x[2] / T_f + F_fy_stat / T_f
    F_ry_dot = -x[3] / T_r + F_ry_stat / T_r

    return np.array([beta_dot, yaw_rate_dot, F_fy_dot, F_ry_dot])

def h(x, u):
    ay = (1 / m) * (x[2] * np.cos(u[0]) + x[3])
    return np.array([ay, x[1]])

def ekf_predict(x, P, u, F_fy_stat, F_ry_stat):

    # F_k = np.array([
    #     [(1/(m*u[1]))*(x[2]*np.sin(u[0]-x[0])-x[3]*np.sin(x[0])), -1, np.cos(u[0]-x[0])/(m*u[1]), np.cos(x[0])/(m*u[1])],
    #     [0, 0, (np.cos(u[0])*l_f)/I_zz, -l_r/I_zz],
    #     [0, 0, -1/T_f, 0],
    #     [0, 0, 0, -1/T_r]
    # ])

    # ((x[2]*np.sin(u[0]-x[0]))/(m*u[1]))-((x[3]*np.sin(x[0]))/(m*u[1]))
    # (1/(m*u[1]))*(x[2]*np.sin(u[0]-x[0])-x[3]*np.sin(x[0]))

    F_k = np.eye(4)

    x_pred = x + f(x, u, F_fy_stat, F_ry_stat) * dt
    P_pred = F_k @ P @ F_k.T + Q
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, z, u):
    H_k = np.array([[0, 0, np.cos(u[0])/m, 1/m], [0, 1, 0, 0]])

    S = H_k @ P_pred @ H_k.T + R
    K = P_pred @ H_k.T @ np.linalg.inv(S)

    y = z - h(x_pred, u)

    x_upd = x_pred + K @ y
    P_upd = (np.eye(4) - K @ H_k) @ P_pred
    return x_upd, P_upd


# 5. 랙 포스 추정
def high_speed_rack_force(x):
    F_move = x[2] * (t_m + t_p) * i_p
    return F_move

def stat_rack_force(theta, delta_dot):

    if np.sign(theta) == np.sign(delta_dot):
        theta_dot = delta_dot * (1 - (c_theta * abs(theta) / Mz_max) ** 2)
    else:
        theta_dot = delta_dot

    theta = theta + theta_dot * dt
    rack_force = c_theta * theta * i_p

    return rack_force, theta
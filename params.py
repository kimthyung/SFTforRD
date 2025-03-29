import numpy as np


v_threshold = 7/3.6
########### #################
######### High Speed ############

m = 1875.05  # 차량 질량 (kg)
I_zz = 3236.858  # z축 관성 모멘트 (kg*m^2)
l_f = 1.450  # 차량 앞바퀴 중심까지의 거리 (m)
l_r = 1.450  # 차량 뒷바퀴 중심까지의 거리 (m)
h_cg = 0.589    #height of center of gravity (m)
mu = 1.0  # 횡방향 도로 마찰 계수

K = 25
t_m = 0.025 
t_p = 0.2
i_p = -1

# 시간 간격
dt = 0.01
g = 9.81

#delay constant
T_f = 0.02
T_r = 0.03

################################
######## Stationary ############
c_theta = 1500
theta = 0    
theta_dot = 0   
Mz_max = 50


Q = np.diag([2, 0.01, 0.5, 0.1])
R = np.diag([0.1, 0.01]) 
P = np.diag([1,1,5,5])

# 상태 초기값 및 공분산 초기화
x = np.array([0, 0, 0, 0])  # [beta, yaw_rate, F_fy, F_ry]
u = np.array([0, 0])  # [delta, v]

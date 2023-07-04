import scientific_research_with_python_demo.scientific_research_with_python_demo.utils as af
import numpy as np

WAVELENGTH = 0.0056  # [unit:m]
Nifg = 3
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 100
param_name = ["height", "velocity"]
normal_baseline = np.random.normal(size=(1, Nifg)) * 333
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
m2ph = 4 * np.pi / WAVELENGTH
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
print(h2ph)
# 合并两个数组v2ph和h2ph
par2ph = np.concatenate((h2ph, v2ph), axis=1) * m2ph
print(par2ph.shape)
print(par2ph)
a_mat = 2 * np.pi * np.eye(Nifg)
# 合并a_mat和par2ph
A_1 = np.concatenate((a_mat, par2ph), axis=1)
print(A_1.shape)
print(A_1)
P = np.concatenate((np.zeros((2, Nifg)), np.eye(2)), axis=1)
print(P.shape)
# 合并par2ph和P
A = np.concatenate((A_1, P), axis=0)
print(A.shape)
print(A)
# print(A[19][19])
phase_obs, srn, phase_true = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
print(phase_true)
print(phase_obs)
y_00 = np.array([[30, 0.05]]).T
y_01 = np.array([[29.99, 0.0499]]).T
# 合并phase_obs和y_0
y_1 = np.concatenate((phase_obs, y_00), axis=0)
y_2 = np.concatenate((phase_obs, y_01), axis=0)
x = np.linalg.inv(A).dot(y_1)
print(x)

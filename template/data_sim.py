import scientific_research_with_python_demo.scientific_research_with_python_demo.utils as af
import numpy as np

WAVELENGTH = 0.0056  # [unit:m]
Nifg = 50
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 70
param_name = ["height", "velocity"]
normal_baseline = np.random.normal(size=(1, Nifg)) * 333
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
# print(h2ph)
par2ph = [h2ph, v2ph]
par2ph1 = np.hstack((h2ph, v2ph))
a_mat = 2 * np.pi * np.eye(Nifg)
A = np.hstack((par2ph1, a_mat))
phase_obs = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph)
# simulate noise phase
noise_phase = af.gauss_noise(phase_obs, noise_level)
phase_obs += noise_phase
# 使用lstsq函数计算最小二乘解
coef, residuals, rank, singular_values = np.linalg.lstsq(A, phase_obs, rcond=None)

# 输出最小二乘解
print(coef)
# print(par2ph)
print(A.shape)
# 最小二乘法估计

import scientific_research_with_python_demo.utils as af
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
# 使用stack合并两个数组v2ph和h2ph
par2ph = np.hstack((h2ph, v2ph)) * m2ph
print(par2ph.shape)
print(par2ph)
# par2ph = np.concatenate((h2ph, v2ph), axis=1) * m2ph
# a_mat = 2 * np.pi * np.eye(Nifg)
phase_obs, srn, phase_true = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
print(phase_true)
print(phase_obs)

import scientific_research_with_python_demo.utils as af
import numpy as np

WAVELENGTH = 0.056  # [unit:m]
Nifg = 20
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
pseudo_obs = np.array([[28, 0.04]]).T
noise_level = 100
param_name = ["height", "velocity"]
normal_baseline = np.random.randn(1, Nifg) * 333
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
m2ph = 4 * np.pi / WAVELENGTH
std_param = np.array([40, 0.06])
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
sig2 = (np.pi * 30 / 180) ** 2 * np.ones(Nifg)
# print(h2ph)
# print(v2ph)
Q_y = af.cov_obs(sig2, std_param)
print(Q_y)
phase_obs, srn, phase_true = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
a_true = (phase_obs - phase_true) / (2 * np.pi)
print(a_true)
print(phase_true)
# print(phase_obs)
A_desin, y = af.design_mat(h2ph, v2ph, phase_obs, pseudo_obs)
print(A_desin)
# a_hat = y / (2 * np.pi)
a_hat = af.compute_ahat(A_desin, y)
print(a_hat)
# print(y[0:5, :] / (2 * np.pi))
# print(y)
Q_ahat = af.cov_ahat(A_desin, Q_y, Nifg)
print(Q_ahat)
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/a_hat1.csv", a_hat, delimiter=",")
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Q_ahat1.csv", Q_ahat, delimiter=",")
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/a_true1.csv", a_true, delimiter=",")

import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_main import periodogram
import scientific_research_with_python_demo.data_plot as dp
import numpy as np

WAVELENGTH = 0.056  # [unit:m]
Nifg = 2
v_orig = 2  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 1  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 70
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 1])
std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]

# calculate the number of search
Num_search1_max = 1  # Num_search1 for height
Num_search1_min = 1
Num_search2_max = 1  # Num_search2 for velocity
Num_search2_min = 1
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
normal_baseline = np.array([[1, 2]]) * (af.R * np.sin(af.Incidence_angle))
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg) * 365 / 12
v2ph = af.v_coef(time_baseline).T / af.m2ph
h2ph = af.h_coef(normal_baseline).T / af.m2ph
par2ph = [h2ph, v2ph]
phase_obs, snr, phase_true = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
print(phase_obs)
print(phase_true)
# data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name)
# est_param = {}
# est_param, best = periodogram(data_set, phase_obs)
# coh = np.array(
#     [
#         [
#             -0.27739318 - 0.7506298j,
#             -0.39826176 + 0.11589664j,
#             -0.01522649 - 0.07061029j,
#             -0.39826176 + 0.11589664j,
#             -0.01522649 - 0.07061029j,
#             -0.5361442 + 0.0764255j,
#             -0.01522649 - 0.07061029j,
#             -0.5361442 + 0.0764255j,
#             0.06212853 + 0.87610056j,
#         ]
#     ]
# )
# a = abs(coh)
# print(abs(coh))
# print(af.find_maximum_coherence(coh))
# print(est_param)
signal = np.array([[1, 1, 1, 1, 1]]).T
noise = af.gauss_noise(signal, np.pi * 20 / 180)
signal_noise = signal + noise
snr = af.check_snr2(signal, noise)
print(signal_noise)
print(snr)

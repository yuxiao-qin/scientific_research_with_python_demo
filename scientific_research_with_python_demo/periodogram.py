import numpy as np
from scientific_research_with_python_demo.main import v2phase, h2phase, sim_phase_noise, sim_arc_phase, search_parm_solution, maximum_temporal_coh


# Initialize

v_orig = 0.05  # [mm/year]
h_orig = 30  # [m]
noise_level = 0.0
# normal_baseline = np.random.normal(size=(1, 20))*300
normal_baseline = np.array([[-235.25094786, -427.79160933, 36.37235105, 54.3278281, -87.27348344,
                             25.31470275, 201.85998322, 92.22902115, 244.66603228, -89.80792772,
                             12.17022031, -23.71273067, -241.58736045, -184.03477855, - 15.97933883,
                             -116.39428378, -545.53546226, -298.89492777, -379.2293736, 289.30702061]])

time_range = np.arange(1, 21, 1).reshape(1, 20)
# phase_obsearvation simulate
phase_orig = sim_arc_phase(v_orig, h_orig, noise_level,
                           time_range, normal_baseline)
# print(normal_baseline)
# print(time_range)
# print(phase_orig)
# phase_orig = np.array([[-0.16197983, 2.07703957, -0.38777374,  0.6686454,  0.69867534, 2.14699018,
#                         -2.25000165, -2.00269921, 1.26480047, -0.22241084, -0.93141071, 1.744535,
#                         -1.16754325, 1.65687907, 1.8168527, -2.34515856, 2.05190303, -0.65915225,
#                         -0.73824169, 0.65241122]])
# # print(phase_orig.shape)

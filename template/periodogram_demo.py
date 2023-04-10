import os, sys

sys.path.append((os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))))
import scientific_research_with_python_demo.utils as af
import scientific_research_with_python_demo.periodogram_main as periodogram
import numpy as np

# import os, sys

# sys.path.append(
#     "C:\\Users\\Administrator\\Desktop\\scientific_research_with_python_demo\\scientific_research_with_python_demo"
# )

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.0056  # [unit:m]
v_orig = 0.05  # [mm/year]
h_orig = 30  # [m]
noise_level = 0.0
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.03])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]

# calculate the number of search
Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
Num_search = np.array([Num_search1, Num_search2])
# simulate baseline
normal_baseline = np.random.normal(size=(1, 20)) * 300
time_baseline = np.arange(1, 21, 1).reshape(1, 20)
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
par2ph, search_num, step, param = af.input_parameters(v2ph, h2ph, Num_search, step_orig, param_orig)

# phase_obsearvation simulate
phase_obs = af.sim_arc_phase(v_orig, h_orig, noise_level, v2ph, h2ph)

# ------------------------------------------------
# main loop of searching
# ------------------------------------------------
count = 0
while count <= 1:
    # search the parameters
    est_param = periodogram(par2ph, phase_obs, search_num, step, param)
    # update the parameters
    param = est_param
    # update the step
    step_orig *= 0.1
    step = af.list2dic(step_orig, param_name)
    # update the number of search
    Num_search = np.array([20, 20])
    search_num = af.list2dic(Num_search, param_name)

    count += 1
print(est_param)


# normal_baseline = np.array(
#     [
#         [
#             -235.25094786,
#             -427.79160933,
#             36.37235105,
#             54.3278281,
#             -87.27348344,
#             25.31470275,
#             201.85998322,
#             92.22902115,
#             244.66603228,
#             -89.80792772,
#             12.17022031,
#             -23.71273067,
#             -241.58736045,
#             -184.03477855,
#             -15.97933883,
#             -116.39428378,
#             -545.53546226,
#             -298.89492777,
#             -379.2293736,
#             289.30702061,
#         ]
#     ]
# )

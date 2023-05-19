import scientific_research_with_python_demo.scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.scientific_research_with_python_demo.periodogram_main import periodogram
import numpy as np

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
v = np.linspace(0.001, 0.01, 10)
# v = 0.05
h = np.linspace(30, 40, 10)
# h = 30
WAVELENGTH = 0.0056  # [unit:m]
Nifg = 20
noise_level = 0.0
noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]

# calculate the number of search
Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
Num_search = np.array([Num_search1, Num_search2])

# simulate baseline
# normal_baseline = np.random.normal(size=(1, Nifg)) * 300
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
normal_baseline = np.array(
    [
        [
            -235.25094786,
            -427.79160933,
            36.37235105,
            54.3278281,
            -87.27348344,
            25.31470275,
            201.85998322,
            92.22902115,
            244.66603228,
            -89.80792772,
            12.17022031,
            -23.71273067,
            -241.58736045,
            -184.03477855,
            -15.97933883,
            -116.39428378,
            -545.53546226,
            -298.89492777,
            -379.2293736,
            289.30702061,
        ]
    ]
)
# calculate the input parameters of phase
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
# print(h2ph)
par2ph = [h2ph, v2ph]
# lab loop
for i in range(10):
    # print(time_baseline)
    # v_orig = v[i]  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
    v_orig = v[i]
    for j in range(10):
        h_orig = h[j]  # [m]，整数 30 循环迭代搜索结果有问题
        # h_orig = h

        # phase_obsearvation simulate
        phase_obs = af.sim_arc_phase(v_orig, h_orig, noise_level, v2ph, h2ph, noise_phase)
        # print(phase_obs)
        # normalize the intput parameters
        data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name)
        # print(data_set)
        # print(data_set["velocity"]["Num_search"])
        # est_param = periodogram(data_set, phase_obs)
        # print(est_param)
        # ------------------------------------------------
        # main loop of searching
        # ------------------------------------------------
        count = 0
        est_param = {}
        while count <= 10 and data_set["velocity"]["step_orig"] > 1.0e-8 and data_set["height"]["step_orig"] > 1.0e-4:
            # search the parameters
            est_param, best = periodogram(data_set, phase_obs)
            # update the parameters
            for key in param_name:
                data_set[key]["param_orig"] = est_param[key]
                # update the step
                data_set[key]["step_orig"] *= 0.1
                # update the number of search
                data_set[key]["Num_search"] = 20

            count += 1
        print({"h_set": h_orig, "v_set": v_orig, "Nifg": Nifg, "dt": 12})
        print(est_param)
    # ambiguty solution
    # ambiguities = af.ambiguity_solution(data_set, 1, best, phase_obs)
    # print(ambiguities)
    # print(data_set)

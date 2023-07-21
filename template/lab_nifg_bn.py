import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_main import periodogram
import numpy as np
import time
import scientific_research_with_python_demo.data_plot as dp

# Nifg 的实验
T1 = time.perf_counter()

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.0056  # [unit:m]
Nifg_orig = np.linspace(10, 50, 40, dtype=int)
v_orig = 0.005  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 10
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.01])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]

# calculate the number of search
Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
Num_search = np.array([Num_search1, Num_search2])

success_rate = np.zeros(len(Nifg_orig))
est_height = np.zeros(len(Nifg_orig))
est_velocity = np.zeros(len(Nifg_orig))
for i in range(len(Nifg_orig)):
    iteration = 0
    success = 0
    Nifg = Nifg_orig[i]
    # print("Nifg = ", Nifg)
    # simulate baseline
    normal_baseline = np.random.normal(size=(1, Nifg)) * 333
    # print(normal_baseline)
    time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
    # print(time_baseline)
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
    # calculate the input parameters of phase
    v2ph = af.v_coef(time_baseline).T
    h2ph = af.h_coef(normal_baseline).T
    # print(h2ph)
    par2ph = [h2ph, v2ph]
    # phase_obsearvation simulate
    phase_obs = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph)
    # simulate noise phase
    noise_phase = af.gauss_noise(phase_obs, noise_level)
    phase_obs += noise_phase
    # normalize the intput parameters
    data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name)
    # print(data_set)
    # print(data_set["velocity"]["Num_search"])
    # ------------------------------------------------
    # main loop of searching
    # ------------------------------------------------
    count = 0
    est_param = {}
    while count <= 1 and data_set["velocity"]["step_orig"] > 1.0e-8 and data_set["height"]["step_orig"] > 1.0e-4:
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
    est_height[i] = est_param["height"]
    est_velocity[i] = est_param["velocity"]
    # if abs(est_param["height"] - h_orig) < 0.5 and abs(est_param["velocity"] - v_orig) < 0.005:
    #     success += 1
    # else:
    #     print(est_param)
print(est_height)
print(est_velocity)
# bar_plot(Nifg_orig, success_rate, "Nifg1", 10)
# print(success_rate)
desired = h_orig
dp.scatter_plot(est_height, desired, "Nifg", "h/m", Nifg_orig, "SNR=10db", "demo23")
dp.scatter_plot(est_velocity, 0.005, "Nifg", "v/mm/year", Nifg_orig, "SNR=10db", "demo24")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))

import scientific_research_with_python_demo.scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.scientific_research_with_python_demo.periodogram_main import periodogram
import scientific_research_with_python_demo.scientific_research_with_python_demo.data_plot as dp
import numpy as np
import time

# Nifg 的实验
T1 = time.perf_counter()

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.0056  # [unit:m]
Nifg = 20
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 70
dt_orig = np.linspace(0.1, 1.0, 10)
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
# std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
v = 0.05
# # calculate the number of search
# Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
# Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
# Num_search = np.array([Num_search1, Num_search2])
std_param = np.array([40, 0.08])
# calculate the number of search
Num_search1_max = 80
Num_search1_min = 80
Num_search2_max = 1600
Num_search2_min = 1600
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
success_rate = np.zeros(len(dt_orig))
for i in range(len(dt_orig)):
    dt = dt_orig[i]
    print("dt = ", dt)
    iteration = 0
    success = 0
    while iteration < 100:
        iteration += 1
        # simulate baseline
        normal_baseline = np.random.normal(size=(1, Nifg)) * 333
        # print(normal_baseline)
        time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg) * dt  # 减小重访周期 dt 能明显改善结果
        # print(time_baseline)
        # calculate the input parameters of phase
        v2ph = af.v_coef(time_baseline).T
        h2ph = af.h_coef(normal_baseline).T
        # print(h2ph)
        par2ph = [h2ph, v2ph]
        # phase_obsearvation simulate
        phase_obs = af.sim_arc_phase(v, h_orig, v2ph, h2ph)
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
        while count <= 2 and data_set["velocity"]["step_orig"] > 1.0e-8 and data_set["height"]["step_orig"] > 1.0e-4:
            # search the parameters
            est_param, best = periodogram(data_set, phase_obs)
            # update the parameters
            for key in param_name:
                data_set[key]["param_orig"] = est_param[key]
                # update the step
                data_set[key]["step_orig"] *= 0.1
                # update the number of search
                data_set[key]["Num_search_max"] = 10
                data_set[key]["Num_search_min"] = 10

            count += 1
        if abs(est_param["height"] - h_orig) < 0.5 and abs(est_param["velocity"] - v) < 0.005:
            success += 1
        # else:
        #     print(est_param)
    # success rate
    print(success / iteration)
    success_rate[i] = success / iteration
print(success_rate)
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt_20_1600_1600.txt", success_rate)
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
dp.bar_plot(dt_orig * 12, success_rate, "Nifg=20,SNR=70db", "dt_20_1600_1600", 0.1 * 12, "v[mm/year]")

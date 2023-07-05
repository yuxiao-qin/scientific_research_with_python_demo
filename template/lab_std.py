import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_main import periodogram
import scientific_research_with_python_demo.data_plot as dp
import numpy as np
import time

# Nifg 的实验
T1 = time.perf_counter()

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.0056  # [unit:m]
Nifg = 20
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 50
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
std_v = np.linspace(0, 0.1, 51)
# std_v = [0.1]
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
v = v_orig
h = h_orig
# simulate baseline
normal_baseline = np.fromfile(
    "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline20.bin", dtype=np.float64
).reshape(1, Nifg)
iteration = 0
success = 0
est_velocity = np.zeros(51)
# std_param = {"height": 40, "velocity": 0.1}

while iteration < 51:
    std_param = np.array([100, std_v[iteration]])
    # print(std_param)
    Num_search1_max = 80
    Num_search1_min = 80
    Num_search2_max = 1600
    Num_search2_min = af.compute_Nsearch(std_param[1], step_orig[1])
    Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
    print(Num_search)
    # print(normal_baseline)
    time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
    # print(time_baseline)
    # calculate the input parameters of phase
    v2ph = af.v_coef(time_baseline).T
    h2ph = af.h_coef(normal_baseline).T
    # print(h2ph)
    par2ph = [h2ph, v2ph]
    # phase_obsearvation simulate
    phase_obs, snr = af.sim_arc_phase(v, h, v2ph, h2ph, noise_level)
    # print(phase_obs.shape)
    # print(phase_obs)
    # normalize the intput parameters
    data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name)
    # print(data_set["velocity"]["Num_search_max"], data_set["velocity"]["Num_search_min"])
    # print(data_set)
    # print(data_set["velocity"]["Num_search"])
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
            data_set[key]["step_orig"] /= 10
            # update the number of search
            data_set[key]["Num_search_max"] = 10
            data_set[key]["Num_search_min"] = 10
        # print(data_set["velocity"]["step_orig"], data_set["height"]["step_orig"])
        count += 1
    print(est_param)
    if abs(est_param["height"] - h) < 0.01 and abs(est_param["velocity"] - v) < 0.00014:
        success += 1
    est_velocity[iteration] = est_param["velocity"]
    iteration += 1
    # else:
    #     print(est_param)
# success rate
print(success / iteration)
print(est_velocity)
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/std0_0_0_1_02.txt", est_velocity)
# dp.hist_plot(est_velocity, "std0_0_0_1_02", "v/m/year", "count", 100, "Nifg=20,SNR=70db,v=0.05")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))

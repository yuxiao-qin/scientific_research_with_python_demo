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
Nifg = 10
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 70
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]

# calculate the number of search
Num_search1_max = 200  # Num_search1 for height
Num_search1_min = 80
Num_search2_max = 1300  # Num_search2 for velocity
Num_search2_min = 300
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
iteration = 0
success = 0
est_velocity = np.zeros(100)
# std_param = {"height": 40, "velocity": 0.1}
while iteration < 1:
    # simulate baseline
    normal_baseline = np.random.normal(size=(1, Nifg)) * 333
    # print(normal_baseline)
    # normal_baseline.tofile("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline50.bin")
    # print(normal_baseline)
    # normal_baseline = np.fromfile(
    #     "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline20.bin", dtype=np.float64
    # ).reshape(1, Nifg)
    time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
    # print(time_baseline)
    # calculate the input parameters of phase
    v2ph = af.v_coef(time_baseline).T
    h2ph = af.h_coef(normal_baseline).T
    # print(h2ph)
    par2ph = [h2ph, v2ph]
    print(par2ph)
    # phase_obsearvation simulate
    phase_obs, snr, phase_true = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph, noise_level)
    # print(snr)
    # print(phase_obs)
    # normalize the intput parameters
    data_set = af.input_parameters(par2ph, step_orig, Num_search, param_orig, param_name)
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
            data_set[key]["step_orig"] *= 0.1
            # update the number of search
            data_set[key]["Num_search_max"] = 10
            data_set[key]["Num_search_min"] = 10
        count += 1
    # print(est_param)
    if abs(est_param["height"] - h_orig) < 0.01 and abs(est_param["velocity"] - v_orig) < 0.00014:
        success += 1
        print(est_param)
    est_velocity[iteration] = est_param["velocity"]
    iteration += 1
    # else:
# success rate
print(success / 100)
# print(est_param)
# print(est_velocity)
# np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/est_velocity.txt", est_velocity)
# dp.hist_plot(est_velocity, "demo28", "time", "count", 10, "hist")
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))

# ambiguty solution
# ambiguities = af.ambiguity_solution(data_set, 1, best, phase_obs)
# print(ambiguities)
# print(data_set)

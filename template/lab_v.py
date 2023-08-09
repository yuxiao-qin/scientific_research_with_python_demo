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
WAVELENGTH = 0.056  # [unit:m]
Nifg = 20
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 50
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
# std_param = np.array([40, 0.06])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
# v_orig = np.linspace(0, 0.2, 50, dtype=np.float32)  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
# v_orig = np.linspace(0, 0.1, 50)
# v_orig = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04])
# v_orig = np.linspace(0.01, 0.1, 10)
v_orig = np.arange(1, 201, 1) * 0.001
h = h_orig
# # calculate the number of search
# Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
# Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
# Num_search = np.array([Num_search1, Num_search2])
std_param = np.array([40, 0.08])
# calculate the number of search
Num_search1_max = 120
Num_search1_min = 120
Num_search2_max = 1600
Num_search2_min = 1600
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
success_rate = np.zeros(len(v_orig))
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)
for i in range(len(v_orig)):
    v = v_orig[i]
    print("v = ", v)
    iteration = 0
    success = 0
    est = np.zeros((100, 2))
    while iteration < 100:
        # simulate baseline
        normal_baseline = np.random.normal(size=(1, Nifg)) * 333
        # print(normal_baseline)
        # calculate the input parameters of phase
        v2ph = af.v_coef(time_baseline).T
        h2ph = af.h_coef(normal_baseline).T
        # print(h2ph)
        par2ph = [h2ph, v2ph]
        # phase_obsearvation simulate
        phase_obs, snr, phase_true = af.sim_arc_phase(v, h_orig, v2ph, h2ph, noise_level)
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
        if abs(est_param["height"] - h) < 0.03 and abs(est_param["velocity"] - v) < 0.00005:
            success += 1
        est[iteration, 0] = est_param["height"]
        est[iteration, 1] = est_param["velocity"]
        iteration += 1
        # else:
        # print(est_param)
    with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/V_est_SNR50.csv", "a") as f:
        # 按列追加保存
        np.savetxt(f, est, delimiter=",")
    # success rate
    # print(success / iteration)
    print(success / iteration)
    success_rate[i] = success / iteration
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Vsuccess_SNR50nifg_20.csv", success_rate)
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))
# dp.bar_plot(v_orig * 1000, success_rate, "Nifg=10,SNR=70db,dt=12", "snr_v_test5", 0.001 * 1000, "v[mm/year]")
dp.line_plot(v_orig * 1000, success_rate, "SNR=50,dt=12,v=0.005,h=30", "V_50nifg_20", "v[mm/year]")

# ambiguty solution
# ambiguities = af.ambiguity_solution(data_set, 1, best, phase_obs)
# print(ambiguities)
# print(data_set)

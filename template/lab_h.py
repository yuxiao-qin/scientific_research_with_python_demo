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
v_orig = 0.005  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = np.arange(1, 1501, 1) *0.1 # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 70
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
# calculate the number of search
# Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
# Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
# Num_search = np.array([Num_search1, Num_search2])
# iteration = 0
# success = 0
# std_param = {"height": 40, "velocity": 0.1}
std_param = np.array([100, 0.03])
Num_search1_max = 120
Num_search1_min = 120
Num_search2_max = 1600
Num_search2_min = 1600
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
v = v_orig
success_rate = np.zeros(len(h_orig))
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
for i in range(len(h_orig)):
    h = h_orig[i]
    print("h = ", h)
    est = np.zeros((100, 2))
    # calculate the number of search
    iteration = 0
    success = 0
    while iteration < 100:
        # simulate baseline
        normal_baseline = np.random.normal(size=(1, Nifg)) * 333
        # print(normal_baseline)

        # print(time_baseline)
        # calculate the input parameters of phase
        v2ph = af.v_coef(time_baseline).T
        h2ph = af.h_coef(normal_baseline).T
        # print(h2ph)
        par2ph = [h2ph, v2ph]
        # phase_obsearvation simulate
        phase_obs, snr, phase_true = af.sim_arc_phase(v, h, v2ph, h2ph, noise_level)
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
                data_set[key]["Num_search_max"] = 100
                data_set[key]["Num_search_min"] = 100

            count += 1
        # print(est_param)

        if abs(est_param["height"] - h) < 0.03 and abs(est_param["velocity"] - v) < 0.00005:
            success += 1

        est[iteration, 0] = est_param["height"]
        est[iteration, 1] = est_param["velocity"]
        iteration += 1
    with open("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/TH0.1_est_SNR70nifg20.csv", "a") as f:
        # 按列追加保存
        np.savetxt(f, est, delimiter=",")
    # success rate
    success_rate[i] = success / iteration
    print(success / iteration)
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/TH0.1success_SNR70nifg_20.csv", success_rate)

T2 = time.perf_counter()
dp.line_plot(h_orig, success_rate, "SNR=70,dt=12,v=0.005,nifg=20,loop_search=100dx", "TH0x1_70nifg_20", "h/m")

print("程序运行时间:%s秒" % (T2 - T1))

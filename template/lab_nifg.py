import scientific_research_with_python_demo.scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.scientific_research_with_python_demo.periodogram_main import periodogram
import numpy as np
import time
import scientific_research_with_python_demo.scientific_research_with_python_demo.data_plot as dp

# Nifg 的实验
T1 = time.perf_counter()

# ------------------------------------------------
# initial parameters
# ------------------------------------------------
WAVELENGTH = 0.0056  # [unit:m]
# Nifg_orig = np.linspace(10, 50, 41, dtype=int)
Nifg_orig = [49]
v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
noise_level = 70
# noise_phase = af.sim_phase_noise(noise_level, Nifg)
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.01])
param_orig = np.array([0, 0])
param_name = ["height", "velocity"]
v = v_orig
h = h_orig
# calculate the number of search
Num_search1_max = 80
Num_search1_min = 80
Num_search2_max = 1600
Num_search2_min = 300
Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
success_rate = np.zeros(len(Nifg_orig))
est = np.zeros((81, 100))
for i in range(len(Nifg_orig)):
    iteration = 0
    success = 0
    Nifg = Nifg_orig[i]
    print("Nifg = ", Nifg)
    while iteration < 100:
        # simulate baseline
        normal_baseline = np.random.normal(size=(1, Nifg)) * 333
        # print(normal_baseline)
        time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
        # print(time_baseline)
        # calculate the input parameters of phase
        v2ph = af.v_coef(time_baseline).T
        h2ph = af.h_coef(normal_baseline).T
        # print(h2ph)
        par2ph = [h2ph, v2ph]
        # phase_obsearvation simulate
        # phase_obsearvation simulate
        phase_obs, snr = af.sim_arc_phase(v, h, v2ph, h2ph, noise_level)
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
                data_set[key]["Num_search"] = 10

            count += 1
        if abs(est_param["height"] - h) < 0.5 and abs(est_param["velocity"] - v) < 0.00025:
            success += 1
        est[i, iteration] = est_param["height"]
        est[i + 40, iteration] = est_param["velocity"]
        iteration += 1
    # success rate
    print(success / iteration)
    est[80][i] = success / iteration
np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/Nifg_70.csv", est, delimiter=",")

dp.bar_plot(Nifg_orig, success_rate, "SNR=70,dt=12,v=0.05,h=30", "Nifg_70", 1, "Nifg")
print(success_rate)
T2 = time.perf_counter()
print("程序运行时间:%s秒" % (T2 - T1))

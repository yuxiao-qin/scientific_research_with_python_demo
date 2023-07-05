import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_main import periodogram
import scientific_research_with_python_demo.data_plot as dp
import numpy as np


def est_af(Nifg, v_orig, h_orig, noise_level, step_orig, std_h, std_v, param_orig, param_name):
    std_param = np.array([std_h, std_v])
    # calculate the number of search
    Num_search1_max = af.compute_Nsearch(std_param[0], step_orig[0])
    Num_search1_min = Num_search1_max
    Num_search2_max = af.compute_Nsearch(std_param[1], step_orig[1])
    Num_search2_min = Num_search2_max
    Num_search = np.array([[Num_search1_max, Num_search1_min], [Num_search2_max, Num_search2_min]])
    iteration = 0
    success = 0
    est_all = np.zeros((100, 2))
    while iteration < 100:
        # simulate baseline
        normal_baseline = np.random.normal(size=(1, Nifg)) * 333
        # print(normal_baseline)
        normal_baseline.tofile("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline.bin")
        # print(normal_baseline)
        time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)  # 减小重访周期 dt 能明显改善结果
        # print(time_baseline)

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
        # print(est_param)
        if abs(est_param["height"] - h_orig) < 0.5 and abs(est_param["velocity"] - v_orig) < 0.005:
            success += 1

        est_all[iteration, 0] = est_param["height"]
        est_all[iteration, 1] = est_param["velocity"]
        iteration += 1
    # else:
    success_rate = success / iteration
    # success rate
    print(success / iteration)
    print(est_all)
    return est_all, success_rate
    # print(est_velocity)
    # np.savetxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/est_velocity.txt", est_velocity)
    # dp.hist_plot(est_velocity, "demo28", "time", "count", 10, "hist")

    # ambiguty solution
    # ambiguities = af.ambiguity_solution(data_set, 1, best, phase_obs)
    # print(ambiguities)
    # print(data_set)

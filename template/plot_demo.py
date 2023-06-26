import scientific_research_with_python_demo.scientific_research_with_python_demo.data_plot as dp
import numpy as np

# success_rate = [0.83, 0.87, 0.87, 0.86, 0.85, 0.94, 0.95, 0.93, 0.91]
success_rate = [
    0.15,
    0.21,
    0.34,
    0.28,
    0.48,
    0.48,
    0.54,
    0.57,
    0.64,
    0.73,
    0.7,
    0.76,
    0.8,
    0.84,
    0.78,
    0.86,
    0.89,
    0.91,
    0.91,
    0.94,
    0.92,
    0.98,
    0.93,
    0.98,
    0.99,
    0.99,
    1.0,
    1.0,
    0.98,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]
# v_orig = np.linspace(0.001, 0.01, 10, dtype=np.float32)
Nifg = np.linspace(10, 100, 10)
# x = 0.011
# h = np.arange(10, 100, 10)
# print(h)
# dp.bar_plot(v, success_rate, "test0", 0.01)
# dp.bar_plot(h, success_rate, "test2", 10, "Nifg,v=%s" % x)
# dp.bar_plot(h, success_rate, "test2", 10, "h")
# dp.bar_plot(v_orig * 1000, success_rate, "demo20", 0.001 * 1000, "Nifg=30,v[mm/year]")
# dp.bar_plot(h, success_rate, "demo20", 10, "h/m")

est_height = [30.0, 28.0, 32.0, 29.0, 30.0, 32.0, 29.0, 30.0, 31.0, 29.0]
desired = 30
# dp.scatter_plot(est_height, desired, "Nifg", "h/m", Nifg, "Test of Nifg and h", "demo22")
# durations = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/est_velocity.txt", delimiter=",")
# est_velocity = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/std0_05_0_1_03.txt", delimiter=",")
# dp.hist_plot(est_velocity, "std0_05_0_1_03", "v/m/year", "count", 100, "Nifg=20,SNR=70db,v=0.05")
# normal_baseline = np.fromfile(
#     "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline.bin", dtype=np.float64
# ).reshape(1, 50)
# est_velocity = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/est_velocity1.txt", delimiter=",")
# dp.hist_plot(est_velocity, "demo31", "v/m/year", "count", 20, "hist")
# # print(normal_baseline)
# dt = np.linspace(0.1, 1, 10)
# success_rate1 = [0.16, 0.3, 0.28, 0.3, 0.31, 0.32, 0.44, 0.19, 0.04, 0.08]
# dp.bar_plot(dt * 12, success_rate1, "Nifg=10,SNR=30db", "dt_10_30", 0.1 * 12, "dt/day")
dt = np.linspace(0.1, 1, 10) * 12
y1 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_4_1.txt", delimiter=",").reshape(10, 1)
y2 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_4_2.txt", delimiter=",").reshape(10, 1)
y3 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_4_3.txt", delimiter=",").reshape(10, 1)
y4 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_6_1.txt", delimiter=",").reshape(10, 1)
y5 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_6_2.txt", delimiter=",").reshape(10, 1)
y6 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_6_3.txt", delimiter=",").reshape(10, 1)
y7 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_8_1.txt", delimiter=",").reshape(10, 1)
y8 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_8_2.txt", delimiter=",").reshape(10, 1)
y9 = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/dt0_8_3.txt", delimiter=",").reshape(10, 1)
# 拼接数组

# y = np.concatenate((y1, y2, y3), axis=1).T
# print(y[0])
# dp.mutil_line(dt, y, "dt_SNR_20", "dt", "Nifg=20")
dv = np.linspace(0.01, 0.1, 10)
y = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9), axis=1).T

print(y)
dp.mutil_line(dv, y, "dt_v_snr70", "v/mm/year", "Nifg=10,20,30,SNR=70db")

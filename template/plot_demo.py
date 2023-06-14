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
# dp.hist_plot(durations, "demo27", "time", "count", 10, "hist")
normal_baseline = np.fromfile(
    "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline.bin", dtype=np.float64
).reshape(1, 50)
est_velocity = np.loadtxt("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/est_velocity1.txt", delimiter=",")
dp.hist_plot(est_velocity, "demo31", "v/m/year", "count", 20, "hist")
# print(normal_baseline)

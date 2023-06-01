import scientific_research_with_python_demo.scientific_research_with_python_demo.data_plot as dp
import numpy as np

success_rate = [1, 1, 1, 0.53, 1, 0.45, 0.48, 0.36, 0.34]
v = np.linspace(0.01, 0.09, 9, dtype=np.float32)
Nifg = np.arange(10, 100, 10)
x = 0.011
# dp.bar_plot(v, success_rate, "test0", 0.01)
dp.bar_plot(Nifg, success_rate, "test1", 10, "Nifg,v=%s" % x)
print(v)

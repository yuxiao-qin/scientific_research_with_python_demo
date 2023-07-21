import scientific_research_with_python_demo.utils as af
import numpy as np

param = np.array([10, 0.2])
WAVELENGTH = 0.0056
m2ph = 4 * np.pi / WAVELENGTH
A = np.array([[1, 3, 5]])
B = np.array([[2, 4, 6]])
# phase_noise = np.random.normal(loc=0.0, scale=0.1, size=(1, 3))
phase_noise = np.array([[-0.14474566, 0.13589448, -0.03066064]])
phase_ture = m2ph * A.T * param[0] + m2ph * B.T * param[1] + phase_noise.T
phase_obs = af.wrap_phase(phase_ture)
desired = (phase_ture - phase_obs) / (2 * np.pi)
data_set = {"height": {"par2ph": A.T, "param_orig": 9.99}, "velocity": {"par2ph": B.T, "param_orig": 0.199}}
best = 1
actual = af.ambiguity_solution(data_set, 1, best, phase_obs)
print(phase_noise)
print(desired)
print(actual)

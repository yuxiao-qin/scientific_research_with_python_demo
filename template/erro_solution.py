import numpy as np
import scientific_research_with_python_demo.utils as af
from scientific_research_with_python_demo.periodogram_main import periodogram

Nifg = 20
normal_baseline = np.fromfile(
    "/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo/data_save/normal_baseline20.bin", dtype=np.float64
).reshape(1, Nifg)
# print(normal_baseline)
time_baseline = np.arange(1, Nifg + 1, 1).reshape(1, Nifg)
v2ph = af.v_coef(time_baseline).T
h2ph = af.h_coef(normal_baseline).T
h_orig = 30
v_orig = 0.05
noise_level = 70
par2ph = [h2ph, v2ph]
# phase_obsearvation simulate
phase_obs = af.sim_arc_phase(v_orig, h_orig, v2ph, h2ph)
# simulate noise phase
noise_phase = af.gauss_noise(phase_obs, noise_level)
phase_obs += noise_phase
a = (phase_obs - af._coef2phase(h2ph, h_orig) - af._coef2phase(v2ph, v_orig)) / (2 * np.pi)
coh_a = np.sum(np.exp(1j * (phase_obs - af._coef2phase(h2ph, h_orig) - af._coef2phase(v2ph, 0.0499)))) / 20
print(a.T)
print(abs(coh_a))
b = (phase_obs - af._coef2phase(h2ph, h_orig) - af._coef2phase(v2ph, -0.035348)) / (2 * np.pi)
coh_b = np.sum(np.exp(1j * (phase_obs - af._coef2phase(h2ph, h_orig) - af._coef2phase(v2ph, -0.0354)))) / 20
print(b.T)
print(abs(coh_b))
print(np.argmax([abs(coh_a), abs(coh_b)]))

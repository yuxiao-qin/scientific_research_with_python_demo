import main as af
import numpy as np
import sys
sys .path.append(
    '/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo')
# Initialize
WAVELENGTH = 0.0056  # [unit:m]
v_orig = 0.05  # [mm/year]
h_orig = 30  # [m]
noise_level = 0.0
step_orig = np.array([1.0, 0.0001])
std_param = np.array([40, 0.03])
Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
Num_search = np.array([Num_search1, Num_search2])

param_orig = np.array([0, 0])
# normal_baseline = np.random.normal(size=(1, 20))*300
normal_baseline = np.array([[-235.25094786, -427.79160933, 36.37235105, 54.3278281, -87.27348344,
                             25.31470275, 201.85998322, 92.22902115, 244.66603228, -89.80792772,
                             12.17022031, -23.71273067, -241.58736045, -184.03477855, - 15.97933883,
                             -116.39428378, -545.53546226, -298.89492777, -379.2293736, 289.30702061]])

# time_baseline = np.arange(1, 21, 1).reshape(1, 20)
# v2ph = af.v_coef(time_baseline).T
# h2ph = af.h_coef(normal_baseline).T

# phase_obsearvation simulate
# phase_obs = af.sim_arc_phase(v_orig, h_orig, noise_level, v2ph, h2ph)

# print(phase_obs)
# param = af.periodogram(v2ph, h2ph, phase_obs,
#                        Num_search, step_orig, param_orig)
# print(param)
# count = 0
# while count <= 1:
#     param = af.periodogram(v2ph, h2ph, phase_obs,
#                            Num_search, step_orig, param_orig)
#     param_orig = param
#     step_orig *= 0.1
#     Num_search = np.array([40, 100])
#     count += 1

# print(len(param_orig))

# h_search = search_parm_solution(
#     step_orig[0], Num_search[0], h2ph, param_orig[0])[0]
# v_search = search_parm_solution(
#     step_orig[1], Num_search[1], v2ph, param_orig[1])[0]

# # num_phase = [20, 120]
# num_phase = [20, 80]
# phase_model = model_phase(v_search, h_search, num_phase)
# best_coh = sim_temporal_coh(phase_obs, phase_model)
# index = maximum_coh(best_coh[0], num_phase)
# print(h_search.shape)
# print(v_search.shape)
# print(phase_obs)
# print(phase_model.shape)
# print(best_coh)
# print(index)

# print(sim_phase_noise(0.1))
# print(normal_baseline)
# print(time_range)
# print(phase_orig)
# phase_orig = np.array([[-0.16197983, 2.07703957, -0.38777374,  0.6686454,  0.69867534, 2.14699018,
#                         -2.25000165, -2.00269921, 1.26480047, -0.22241084, -0.93141071, 1.744535,
#                         -1.16754325, 1.65687907, 1.8168527, -2.34515856, 2.05190303, -0.65915225,
#                         -0.73824169, 0.65241122]])
# # print(phase_orig.shape)

# data = np.array(
#     [[-0.8658+1.8919j, -0.7861+0.7072j, -0.8658+1.6096j, -0.8658+0.0733j]])
# a, b = af.maximum_coh(data)
# phase = np.array([[1, 2, 1, -2],
#                   [2, -4, -3, 1],
#                   [3, 5, 2, 3]])
# coh_exp = np.exp(1j*phase)
# coh_t = np.sum(coh_exp, axis=0, keepdims=True)
# best, index = af.maximum_coh(coh_t)
# print(b)
# print(coh_t)
# print(best, index)

h2ph = np.array([[3, 4, 5]]).T
v2ph = np.array([[2, 4, 6]]).T
A = np.hstack((h2ph, v2ph))
phase = np.mat([[7, 12, 17]]).T
actual1 = A
actual = af.correct_param(A, phase)
print(actual)
# N = np.mat(np.dot(A.T, A))
# R = np.mat(np.linalg.cholesky(N)).T
# print(N)
# print(R)
# rhs = (R.I)*(((R.T).I)*A.T)
# param_correct = rhs*phase
# print(param_correct)

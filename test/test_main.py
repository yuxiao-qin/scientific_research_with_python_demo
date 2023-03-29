# from scientific_research_with_python_demo.main import v_coef, h_coef, coef2phase, sim_phase_noise, sim_arc_phase, param_search, phase_search, sim_temporal_coh, WAVELENGTH, wrap_phase, maximum_coh, periodogram, Incidence_angle, R, index2sub, compute_Nsearch, correct_h2ph
import pytest
import scientific_research_with_python_demo.main as af
import numpy as np


def test_compute_Nsearch():
    step_orig = np.array([1.0, 0.0001])
    std_param = np.array([40, 0.03])
    Num_search1 = af.compute_Nsearch(std_param[0], step_orig[0])
    Num_search2 = af.compute_Nsearch(std_param[1], step_orig[1])
    assert Num_search1 == 80
    assert Num_search2 == 600


def test_v_coef():
    # construct a simulated case
    simuated_time_range = np.arange(1, 21, 1) * 365 / 12
    actual = af.v_coef(simuated_time_range)
    # desired phase is calculated by hand
    desired = np.arange(1, 21, 1)
    assert np.isclose(actual, desired).all()


def test_h_coef():
    # actual=h2phase(simulated_h)
    normal_baseline = np.random.normal(
        size=(1, 20))
    actual = af.h_coef(normal_baseline*(af.R*np.sin(af.Incidence_angle)))
    desired = normal_baseline
    assert np.isclose(actual, desired).all()


def test_generate_phase_noise():
    noise = af.sim_phase_noise(0.1)
    assert noise.shape == (20, 1)


def test_wrap_phase():
    phase = np.arange(20)
    simulate = abs(af.wrap_phase(phase))
    assert (simulate <= np.pi).all


def test_sim_arc_phase():
    v_orig = 0.05  # [mm/year]
    h_orig = 30  # [m]
    noise_level = 0.0
    time_range = np.arange(1, 21, 1).reshape(1, 20)
    normal_baseline = np.array([[-235.25094786, -427.79160933, 36.37235105, 54.3278281, -87.27348344,
                                 25.31470275, 201.85998322,  92.22902115, 244.66603228, -89.80792772,
                                 12.17022031, -23.71273067, -241.58736045, -184.03477855, -15.97933883,
                                 -116.39428378, -545.53546226, -298.89492777, -379.2293736, 289.30702061]])

    v2ph = af.v_coef(time_range)
    h2ph = af.h_coef(normal_baseline)
    simulated = af.sim_arc_phase(
        v_orig, h_orig, noise_level, v2ph, h2ph)
    actual = np.array([[-0.16197983, 2.07703957, -0.38777374,  0.6686454,  0.69867534, 2.14699018,
                        -2.25000165, -2.00269921, 1.26480047, -0.22241084, -0.93141071, 1.744535,
                        -1.16754325, 1.65687907, 1.8168527, -2.34515856, 2.05190303, -0.65915225,
                        -0.73824169, 0.65241122]])
    assert np.isclose(actual, simulated).all()


def test_param_search():
    param_orig = [0, 0]
    Bn = np.mat(np.array([1]*20)).T
    h2ph = af.h_coef(Bn*(af.WAVELENGTH*af.R*np.sin(af.Incidence_angle)))
    param_space = af.param_search(1, 10, param_orig[0])
    actual = af.phase_search(h2ph, param_space).shape
    assert actual == (20, 20)


def test_sim_temporal_coh():
    dphase = np.array([[1, 1, 1]]).T
    search_space = np.array([[1, 3, 2, 3],
                             [2, 3, 4, 1],
                             [1, 2, 1, 2]])

    simulated = af.sim_temporal_coh(dphase, search_space)

    actual = np.array([[np.exp(0)+np.exp(-1j)+np.exp(0),
                      np.exp(-2j)+np.exp(-2j)+np.exp(-1j), np.exp(-1j)+np.exp(-3j)+np.exp(0), np.exp(-2j)+np.exp(0)+np.exp(-1j)]])/20
    assert np.isclose(actual, simulated).all()
    # assert simulated == actual


def test_maximum():
    phase = np.array([[1, 3, 2, 3],
                      [2, 3, 4, 1],
                      [1, 2, 1, 2]])
    phase_sum = np.sum(phase, axis=0)
    actual = np.array([[4, 8, 7, 6]])
    sim_phase = np.exp(phase)
    simluated = np.sum(sim_phase, axis=0, keepdims=True)  # 防止维度丢失
    actual_phase = np.array([[np.exp(1)+np.exp(2)+np.exp(1),
                            np.exp(3)+np.exp(3)+np.exp(2), np.exp(2)+np.exp(4)+np.exp(1), np.exp(3)+np.exp(1)+np.exp(2)]])
    # assert (phase_sum == actual).all
    # assert (simluated == actual_phase).all
    num_search = [2, 2]
    max_param = af.maximum_coh(simluated)
    best = max_param[0]
    best_index = max_param[1]
    para_index = af.index2sub(best_index, num_search)
    actual = np.exp(2)+np.exp(4)+np.exp(1)
    assert best == actual
    assert best_index == 2
    assert para_index == (0, 1)


def test_argmax_complex_number():
    phase = np.array([[1, 2, 1, -2],
                      [2, -4, -3, 1],
                      [3, 5, 2, 3]])
    coh_exp = np.exp(1j*phase)
    coh_t = abs(np.sum(coh_exp, axis=0, keepdims=True))
    best, index = af.maximum_coh(coh_t)
    actual = abs(np.exp(1j)+np.exp(2j)+np.exp(3j))
    data = np.array(
        [[-0.8658+1.8919j, -0.7861+0.7072j, -0.8658+1.6096j, -0.8658+0.0733j]])
    a, b = af.maximum_coh(data)
    # assert index == 0
    assert (best == actual).all
    assert b == 0


def test_correct_h2ph():
    h2ph = np.array([[1, 3, 4, 2], [1, 2, 1, 4], [1, 6, 2, 3]])
    actual = af.correct_h2ph(h2ph.T, 1)
    desired = 0.7727272727272727

    assert np.isclose(actual, desired).all()


def test_ambiguity_phase():
    v2ph = np.array([1, 2, 3]).T
    h2ph = np.array([1, 2, 3]).T
    param = [1, 2]
    best = 1
    phase_obs = np.array([3, 4, 5]).T*af.m2ph
    phase_ambiguity = np.array([0, -2, -4]).T*af.m2ph
    actual1, actual2 = af.ambiguity_phase(v2ph, h2ph, param, best, phase_obs)
    desired = af.wrap_phase(phase_ambiguity)
    assert np.isclose(actual1, desired).all()


def test_correct_param():
    h2ph = np.array([[3, 4, 5]]).T
    v2ph = np.array([[2, 4, 6]]).T
    A = np.hstack((h2ph, v2ph))
    phase = np.array([[7, 12, 17]]).T
    actual1 = A
    actual = af.correct_param(A, phase)
    desired1 = np.array([[3, 4, 5], [2, 4, 6]]).T
    desired = np.mat([[1, 2]]).T

    assert actual1.all() == desired1.all()
    assert actual.all() == desired.all()


def test_periodogram():
    v_orig = 0.05  # [mm/year]
    h_orig = 30  # [m]
    noise_level = 0.0
    Num_search = [40, 10]
    step_orig = [1, 0.01]
    param_orig = [0, 0]
    normal_baseline = np.array([[-235.25094786, -427.79160933, 36.37235105, 54.3278281, -87.27348344,
                               25.31470275, 201.85998322, 92.22902115, 244.66603228, -89.80792772,
                               12.17022031, -23.71273067, -241.58736045, -184.03477855, - 15.97933883,
                               -116.39428378, -545.53546226, -298.89492777, -379.2293736, 289.30702061]])

    time_baseline = np.arange(1, 21, 1).reshape(1, 20)
    v2ph = af.v_coef(time_baseline).T
    h2ph = af.h_coef(normal_baseline).T

    # phase_obsearvation simulate
    phase_obs = af.sim_arc_phase(v_orig, h_orig, noise_level, v2ph, h2ph)
    param = af.periodogram(v2ph, h2ph, phase_obs,
                           Num_search, step_orig, param_orig)
    actual = len(param)
    assert actual == 2

import pytest
import numpy as np
from scientific_research_with_python_demo.main import v2phase, h2phase, sim_phase_noise, sim_arc_phase, search_parm_solution, sim_temporal_coh, WAVELENGTH, wrap_phase, maximum_coh, periodogram


def test_v2phase():
    # construct a simulated case
    simuated_v = 0.1 * WAVELENGTH  # [unit:m/yr]
    simuated_time_range = np.arange(1, 21, 1).reshape(1, 20) * 365 / 12
    actual = v2phase(simuated_v, simuated_time_range)[0]
    # desired phase is calculated by hand
    desired = np.arange(1, 21, 1).reshape(1, 20) * 0.1 * 4 * np.pi
    assert np.isclose(actual, desired).all()


def test_h2phase():
    # actual=h2phase(simulated_h)
    normal_baseline = np.random.normal(size=(1, 20))*300
    actual = h2phase(40, normal_baseline)[0]
    assert actual.shape == (1, 20)


def test_generate_phase_noise():
    noise = sim_phase_noise(0.1)
    assert noise.shape == (1, 20)


def test_wrap_phase():
    phase = np.arange(20)
    simulate = abs(wrap_phase(phase))
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

    simulated = sim_arc_phase(
        v_orig, h_orig, noise_level, time_range, normal_baseline)[0]
    actual = np.array([[-0.16197983, 2.07703957, -0.38777374,  0.6686454,  0.69867534, 2.14699018,
                        -2.25000165, -2.00269921, 1.26480047, -0.22241084, -0.93141071, 1.744535,
                        -1.16754325, 1.65687907, 1.8168527, -2.34515856, 2.05190303, -0.65915225,
                        -0.73824169, 0.65241122]])
    assert np.isclose(actual, simulated).all()


def test_construct_param_search():
    param_orig = [0, 0]
    Bn = np.mat(np.array([1]*20)).T
    simulated = search_parm_solution(1, 10, Bn, param_orig[0])[0]
    actual = simulated.shape
    assert actual == (20, 20)


def test_sim_temporal_coh():
    dphase = np.array([[1, 1, 1]]).T
    search_space = np.array([[1, 3, 2, 3],
                             [2, 3, 4, 1],
                             [1, 2, 1, 2]])

    simulated = sim_temporal_coh(dphase, search_space)[0]

    actual = np.array([[np.exp(0)+np.exp(-1)+np.exp(0),
                        np.exp(-2)+np.exp(-2)+np.exp(-1), np.exp(-1)+np.exp(-3)+np.exp(0), np.exp(-2)+np.exp(0)+np.exp(-1)]])
    assert (actual == simulated).all
    # assert simulated == actual


def test_maximum():
    phase = np.array([[1, 3, 2, 3],
                      [2, 3, 4, 1],
                      [1, 2, 1, 2]])
    phase_sum = np.sum(phase, axis=0)
    actual = np.array([[4, 8, 7]])
    sim_phase = np.exp(phase)
    simluated = np.sum(sim_phase, axis=0)
    actual_phase = np.array([[np.exp(1)+np.exp(2)+np.exp(1),
                            np.exp(3)+np.exp(3)+np.exp(2), np.exp(2)+np.exp(4)+np.exp(1), np.exp(3)+np.exp(1)+np.exp(2)]])
    # assert (phase_sum == actual).all
    # assert (simluated == actual_phase).all
    num_search = [2, 2]
    max_param = maximum_coh(simluated, num_search)
    best = max_param[0]
    best_index = max_param[1]
    para_index = max_param[2]
    actual = np.exp(2)+np.exp(4)+np.exp(1)
    assert best == actual
    assert best_index == 2
    assert para_index == (0, 1)


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

    time_range = np.arange(1, 21, 1).reshape(1, 20)
    phase_orig = sim_arc_phase(v_orig, h_orig, noise_level,
                               time_range, normal_baseline)
    phase_obs = phase_orig[0].T
    v2ph = phase_orig[1].T
    h2ph = phase_orig[2].T
    param = periodogram(v2ph, h2ph, phase_obs,
                        Num_search, step_orig, param_orig)
    actual = len(param)
    assert actual == 2

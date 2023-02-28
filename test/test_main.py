import pytest
import numpy as np
from scientific_research_with_python_demo.main import v2phase, h2phase, sim_phase_noise, sim_arc_phase, search_parm_solution, maximum_temporal_coh, WAVELENGTH, wrap_phase


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
        v_orig, h_orig, noise_level, time_range, normal_baseline)
    actual = np.array([[-0.16197983, 2.07703957, -0.38777374,  0.6686454,  0.69867534, 2.14699018,
                        -2.25000165, -2.00269921, 1.26480047, -0.22241084, -0.93141071, 1.744535,
                        -1.16754325, 1.65687907, 1.8168527, -2.34515856, 2.05190303, -0.65915225,
                        -0.73824169, 0.65241122]])
    assert np.isclose(actual, simulated).all()


def test_construct_param_search():
    Bn = np.mat(np.array([1]*20)).T
    simulated = search_parm_solution(1, 20, Bn)
    A = np.mat(np.array([np.linspace(2.0, 40, 20), [1.0]*20]).T)
    step = [0.001, 1]
    Nsearch = [200, 20]
    Search_space1 = np.mat(
        np.arange(-Nsearch[1]*step[1], Nsearch[1]*step[1], step[1]))
    actual = np.dot(A[:, 1], Search_space1)
    assert np.isclose(actual, simulated).all()


def test_maximum():
    dphase = np.mat([[-0.5, 0, 0.5]]).T*np.pi
    search_space = np.mat(
        [[-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]]*3)*np.pi
    row_num = 9
    num_search = [3, 3]
    parm = [0, 0]
    Nsearch = [1, 1]
    step = [1, 2]
    simulated = maximum_temporal_coh(dphase, search_space,
                                     num_search)
    # simulated=[best, parm, a, best_index],a is the index of param_search
    # assert simulated[0] == 1
    # assert simulated[1] == [[-1], [-2]]
    # assert simulated[2] == [[1], [1]]
    # assert simulated[3] == 1
    actual = [np.exp(-0.4j*np.pi)+np.exp(0.1j*np.pi)+np.exp(0.6j*np.pi)]
    assert np.isclose(actual, simulated[0]).all()
    # assert simulated == actual

import pytest
import numpy as np
from scientific_research_with_python_demo.main import v2phase, h2phase, generate_phase_noise, construct_simulated_arc_phase, construct_param_search_space, maximum_coh_temporal, WAVELENGTH


def test_v2phase():
    # construct a simulated case
    simuated_v = 0.1 * WAVELENGTH  # [unit:m/yr]
    simuated_time_range = np.array([0, 1, 2]) * 365 / 12
    actual = v2phase(simuated_v, simuated_time_range)
    # desired phase is calculated by hand
    desired = np.array([0.0, 0.1, 0.2]) * 4 * np.pi

    assert np.isclose(actual, desired).all()


def test_h2phase():
    simulated_h = np.array([10.0, 11.0, 9.0, 8.0, 13.0])
    ture_h = np.array([10]*5)
    # actual=h2phase(simulated_h)
    actual = h2phase(ture_h)
    desired = h2phase(ture_h)
    assert np.isclose(actual, desired).all()


def test_generate_phase_noise():
    noise = generate_phase_noise(0.1)
    assert noise.shape == (1, 20)


def test_wrap_phase():
    v_orig = 0.01
    h_orig = [10.0]*20+np.random.normal(size=(1, 20))
    noise_level = 0.1
    phase_unwrapped = construct_simulated_arc_phase(
        v_orig, h_orig, noise_level)
    assert phase_unwrapped.shape == (1, 20)


def test_construct_param_search():
    Bn = np.mat(np.array([1]*20)).T
    simulated = construct_param_search_space(1, 20, Bn)
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
    simulated = maximum_coh_temporal(dphase, search_space,
                                     row_num, num_search, parm, Nsearch, step)
    # simulated=[best, parm, a, best_index],a is the index of param_search
    # assert simulated[0] == 1
    # assert simulated[1] == [[-1], [-2]]
    # assert simulated[2] == [[1], [1]]
    # assert simulated[3] == 1
    actual = [np.exp(-0.4j*np.pi)+np.exp(0.1j*np.pi)+np.exp(0.6j*np.pi)]
    assert np.isclose(actual, simulated[0]).all()
    # assert simulated == actual

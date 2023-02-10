import pytest
import numpy as np
from scientific_research_with_python_demo.main import v2phase,h2phase,generate_phase_noise,construct_simulated_arc_phase,construct_param_search_space,maximum_main,WAVELENGTH


def test_v2phase():
    # construct a simulated case
    simuated_v = 0.1 * WAVELENGTH # [unit:m/yr]
    simuated_time_range = np.array([0,1,2]) * 365 / 12
    actual = v2phase(simuated_v, simuated_time_range)
    # desired phase is calculated by hand
    desired = np.array([0.0, 0.1, 0.2]) * 4 * np.pi

    assert np.isclose(actual, desired).all()


def test_h2phase():
    simulated_h=np.array([10.0,11.0,9.0,8.0,13.0])
    ture_h=np.array([10]*5)
    # actual=h2phase(simulated_h)
    actual=h2phase(ture_h)
    desired=h2phase(ture_h)
    assert np.isclose(actual,desired).all()
def test_generate_phase_noise():
    noise=generate_phase_noise(0.1)
    assert noise.shape==(1,20)

def test_wrap_phase():
    v_orig=0.01
    h_orig=[10.0]*20+np.random.normal(size=(1,20))
    noise_level=0.1
    phase_unwrapped=construct_simulated_arc_phase(v_orig,h_orig,noise_level)
    assert phase_unwrapped.shape==(1,20)
def test_construct_param_search():
    Bn=np.mat(np.array([1]*20)).T
    simulated=construct_param_search_space(1,20,Bn)
    A = np.mat(np.array([np.linspace(2.0,40,20),[1.0]*20]).T)
    step=[0.001,1]
    Nsearch=[200,20]
    Search_space1=np.mat(np.arange(-Nsearch[1]*step[1],Nsearch[1]*step[1],step[1]))
    actual=np.dot(A[:,1],Search_space1)
    assert  np.isclose(actual, simulated).all()
def test_maximum():
    dphase=np.mat([[1+1j,1+2j,1+3j]]).T
    search_space=np.mat([[1j,2j,3j],[1j,2j,3j],[1j,2j,3j]])
    simulated=maximum_main(dphase,search_space)
    actual=np.exp(3)
    assert simulated==actual
    
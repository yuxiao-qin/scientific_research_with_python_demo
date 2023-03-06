import pytest
import numpy as np
from scientific_research_with_python_demo.main import v2phase, h2phase, WAVELENGTH, R0, INCIDENCE_ANGLE, generate_phase_noise, construct_simulated_arc_phase


def test_v2phase():
    # construct a simulated case
    simuated_v = 0.1 * WAVELENGTH # [unit:m/yr]
    simuated_time_range = np.array([0,1,2]) * 365 / 12
    actual = v2phase(simuated_v, simuated_time_range)
    # desired phase is calculated by hand
    desired = np.array([0.0, 0.1, 0.2]) * 4 * np.pi

    assert np.isclose(actual, desired).all()


def test_h2phase():
    # construct a simulated case
    simuated_h = 0.1 * WAVELENGTH # [unit:m]
    simuated_normal_baseline_range = np.array([0,1,2]) * R0 * np.sin(INCIDENCE_ANGLE * np.pi / 180)
    actual = h2phase(simuated_h, simuated_normal_baseline_range)
    # desired phase is calculated by hand
    desired = np.array([0.0, 0.1, 0.2]) * 4 * np.pi

    assert np.isclose(actual, desired).all()

def test_generate_phase_noise():
    simulated_noise_level = 1.0
    TOLERANCE = 0.3
    noise_length = 30
    actual_noise = generate_phase_noise(simulated_noise_level, noise_length)
    assert actual_noise.shape == (1, noise_length) and (abs(np.mean(actual_noise)) < TOLERANCE) and (abs(np.var(actual_noise) - simulated_noise_level) < TOLERANCE)


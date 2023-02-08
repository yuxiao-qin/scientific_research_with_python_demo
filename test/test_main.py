import pytest
import numpy as np
from scientific_research_with_python_demo.main import v2phase, WAVELENGTH


def test_v2phase():
    # construct a simulated case
    simuated_v = 0.1 * WAVELENGTH # [unit:m/yr]
    simuated_time_range = np.array([0,1,2]) * 365 / 12
    actual = v2phase(simuated_v, simuated_time_range)
    # desired phase is calculated by hand
    desired = np.array([0.0, 0.1, 0.2]) * 4 * np.pi

    assert np.isclose(actual, desired).all()


def test_h2phase():
    pass

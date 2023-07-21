import pytest
import sys

sys.path.append("/data/tests/jiaxing/scientific_research_with_python_demo/scientific_research_with_python_demo")
import scientific_research_with_python_demo.utils as af
import scientific_research_with_python_demo.periodogram_main as pm
import numpy as np


def test_periodogram():
    v_orig = 0.05  # [mm/year]
    h_orig = 30  # [m]
    noise_level = 0.0
    normal_baseline = np.array(
        [
            [
                -235.25094786,
                -427.79160933,
                36.37235105,
                54.3278281,
                -87.27348344,
                25.31470275,
                201.85998322,
                92.22902115,
                244.66603228,
                -89.80792772,
                12.17022031,
                -23.71273067,
                -241.58736045,
                -184.03477855,
                -15.97933883,
                -116.39428378,
                -545.53546226,
                -298.89492777,
                -379.2293736,
                289.30702061,
            ]
        ]
    )

    time_baseline = np.arange(1, 21, 1).reshape(1, 20)
    v2ph = af.v_coef(time_baseline).T
    h2ph = af.h_coef(normal_baseline).T
    param_name = ["height", "velocity"]
    par2ph = af.list2dic(param_name, [h2ph, v2ph])
    Num_search = af.list2dic(param_name, [40, 10])
    step_orig = af.list2dic(param_name, [1, 0.01])
    param_orig = af.list2dic(param_name, [0, 0])
    # phase_obsearvation simulate
    phase_obs = af.sim_arc_phase(v_orig, h_orig, noise_level, v2ph, h2ph)
    param = pm.periodogram(par2ph, phase_obs, Num_search, step_orig, param_orig)
    actual = param["height"]
    assert actual == 30

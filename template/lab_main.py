import scientific_research_with_python_demo.utils as af
import scientific_research_with_python_demo.data_plot as dp
from scientific_research_with_python_demo.est_main import est_af
import numpy as np
import sys


# ------------------------------------------------
# initial parameters
# ------------------------------------------------
# WAVELENGTH = 0.0056  # [unit:m]
# Nifg = 50
# v_orig = 0.05  # [mm/year] 减少v，也可以改善估计结果，相当于减少了重访周期
# h_orig = 30  # [m]，整数 30 循环迭代搜索结果有问题
# noise_level = 70
# # noise_phase = af.sim_phase_noise(noise_level, Nifg)
# step_orig = np.array([1.0, 0.001])
# std_param = np.array([40, 0.06])
# param_orig = np.array([0, 0])
# param_name = ["height", "velocity"]
def temporal_est(input_params):
    data = af.data_prepare(input_params)
    # ------------------------------------------------
    # estimation process
    # ------------------------------------------------
    est_af(data["Nifg_orig"], data["v_orig"], data["h_orig"], data["noise_level"], data["step_orig"], data["std_h"], data["std_v"], data["param_orig"], data["param_name"])


def parse_parms(parms_file):
    if isinstance(parms_file, str):
        try:
            with open(parms_file, "r") as inp:
                try:
                    parms = eval(inp.read())
                except Exception as e:
                    print("Something wrong with parameters file.")
                    raise e
            return parms
        except Exception as e:
            print("Specified parameters file not found.")
            raise e
    elif isinstance(parms_file, dict):
        return parms_file
    else:
        print("Wrong input format.")
        raise RuntimeError


if __name__ == "__main__":
    # load parameters:
    if len(sys.argv) > 1:
        parms = parse_parms(sys.argv[1])
        temporal_est(parms)
    else:
        print("Not enough input arguments!")

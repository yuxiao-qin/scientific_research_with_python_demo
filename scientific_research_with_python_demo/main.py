import numpy as np

WAVELENGTH = 0.0056  # [unit:m]


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """wrap phase to [-pi,pi]
        input: phase
        output:wrap_phase
    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def sim_arc_phase(dv: float, dh: float, noise_level: float, time_range) -> np.ndarray:
    """ simulate phase of arc between two points based on a module(topographic_height + linear_deformation)
        input:  v: defomation rate per year
                time_range
                h:topographic height per arc
        output: arc_phase: simulated observation phases = topographic_phase + deformation_phase + nosie
    """
    dv_phase = v2phase(dv, time_range)[0]
    dh_phase = h2phase(dh)[0]
    noise_phase = sim_phase_noise(noise_level)
    arc_phase = wrap_phase(dv_phase + dh_phase + noise_phase)
    return arc_phase


def v2phase(v: float, time_range) -> np.ndarray:
    """Calculate phase difference from velocity difference (of two points).
        input: v:defomation rate
               time_range:the factor to caclulate temporal baseline
        output: v2phase_coefficeint(temopral baseline)、v2phase_coefficeint*v(deformation_phase)
    """
    temporal_baseline = 12  # [unit:d]
    temporal_samples = temporal_baseline*time_range
    # distance = velocity * days (convert from d to yr because velocity is in m/yr)
    v2phase_coefficeint = 4 * np.pi * temporal_samples / (WAVELENGTH*365)
    return v2phase_coefficeint*v, v2phase_coefficeint  # [unit:rad]


def h2phase(h: float) -> np.ndarray:
    """Calculate phase difference from topographic height (of two points)
        Input: height per acr
        output:h2ph_coefficient、h2ph_coefficient*h（Topographic phase）

        formation:phase_topo = Height_arc*4pi*Bn/Ri*sin(incidence_angle)
    """
    normal_baseline = np.random.normal(
        size=(1, 20))*300    # perpendicular baseline[unit:m]
    baseline_erro = np.random.rand(1, 20)    # error of perpendicular baseline
    err_baseline = normal_baseline+baseline_erro
    H = 780000    # satellite vertical height[m]
    incidence_angle = 23*np.pi/180    # the local incidence angle
    R = H/np.cos(incidence_angle)    # range to the master antenna.
    h2ph_coefficient = 4*np.pi*err_baseline / \
        (WAVELENGTH*R*np.sin(incidence_angle))
    return h2ph_coefficient*h, h2ph_coefficient


def sim_phase_noise(noise_level: float) -> np.ndarray:
    """ simulate phase noise based on constant noise level
        input: noise_level
        output: noise
    """
    noise = np.random.normal(loc=0.0, scale=noise_level,
                             size=(1, 20))*(4*np.pi/WAVELENGTH)
    return noise


def construct_param_search_space(step: float, Nsearch, A_matrix) -> np.ndarray:
    """
    construct estimated parameters' search space
    such as height [-10:1:10]
    input: step:  step for search
           2*Nsearch :  number of paramters we can search
            创建参数搜索区域[-10:1:10]
    output: Search_space_phase
    """
    Search_space = np.mat(np.arange(-Nsearch*step, Nsearch*step, step))
    Search_space_phase = np.dot(A_matrix, Search_space)
    return Search_space_phase


def construct_search_space(phase_search1, phase_serach2, num_serach) -> np.ndarray:
    # the condition that we have two variables (h,v)
    # 把v和h对应的phase_model展开成相同维度均为（Nifgs*v_search_number*h_search_number）
    search_space = np.kron(phase_search1, np.ones(
        1, num_serach[1]))+np.kron(np.ones(1, num_serach[0]), phase_serach2)
    return search_space


def maximum_coh_temporal(dphase, search_space, row_num_serach, num_search, parm, Nsearch, step):
    # coh_phase=phase_observation-phase_model
    coh_phase = dphase*np.ones((1, row_num_serach))-search_space
    coh_t = np.sum(np.exp(1j*coh_phase), axis=0)
    best = np.max(coh_t)
    best_index = np.argmax(coh_t, axis=1)
    # caculate parameters
    a = np.unravel_index(best_index, (num_search[1], num_search[0]), order="C")
    parm = [parm[0]+(a[0]-(Nsearch[0]))*step[0],
            parm[1]+(a[1]-(Nsearch[1]))*step[1]]
    return best, parm, a, best_index


# v_orig = 0.01
# h_orig = [10]*20+np.random.normal(loc=0, scale=1, size=(1, 20))
# noise_level = 0.1
# phase_unwrapped = construct_simulated_arc_phase(v_orig, h_orig, noise_level).T
# print(h_orig)
# print(h2phase(h_orig))
# print(wrap_phase(phase_unwrapped))
# print(generate_phase_noise(0.1))
# def estimate_parameters(constructed_simulated_phase):
#     # TODO: implement this function
#     return est_dv, est_dh
a = h2phase(40)[0]
print(a)
# print(h2phase(40)[0].shape)
# simuated_v = 0.1 * WAVELENGTH  # [unit:m/yr]
# simuated_time_range = np.arange(1, 21, 1).reshape(1, 20) * 365 / 12
# print(type(v2phase(simuated_v, simuated_time_range)))
# print(simuated_time_range)
# print(np.array(np.linspace(1, 20, 20) * 0.1 * 4 * np.pi))

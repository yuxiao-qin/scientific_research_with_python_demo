import numpy as np

# Constant
WAVELENGTH = 0.0056  # [unit:m]
H = 780000    # satellite vertical height[m]
Incidence_angle = 23*np.pi/180    # the local incidence angle
R = H/np.cos(Incidence_angle)    # range to the master antenna.


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    wrap phase to [-pi,pi]

    input: phase

    output:wrap_phase

    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def sim_arc_phase(v: float, h: float, noise_level: float, time_range) -> np.ndarray:
    """ 
    simulate phase of arc between two points based on a module(topographic_height + linear_deformation)

    input:  v: defomation rate per year
            time_range
            h:topographic height per arc

    output: arc_phase: simulated observation phases = topographic_phase + deformation_phase + nosie

    """
    v_phase = v2phase(v, time_range)[0]
    h_phase = h2phase(h)[0]
    noise_phase = sim_phase_noise(noise_level)
    arc_phase = wrap_phase(v_phase + h_phase + noise_phase)

    return arc_phase


def v2phase(v: float, time_range) -> np.ndarray:
    """
    Calculate phase difference from velocity difference (of two points).

    input: 
    v:defomation rate
    time_range:the factor to caclulate temporal baseline

    output: 
    v2phase_coefficeint: temopral baseline
    v2phase_coefficeint*v: deformation_phase

    """
    temporal_baseline = 12  # [unit:d]
    temporal_samples = temporal_baseline*time_range
    # distance = velocity * days (convert from d to yr because velocity is in m/yr)
    v2phase_coefficeint = 4 * np.pi * temporal_samples / (WAVELENGTH*365)

    return v2phase_coefficeint*v, v2phase_coefficeint  # [unit:rad]


def h2phase(h: float) -> np.ndarray:
    """
    Calculate phase difference from topographic height (of two points)

    Input: height per acr

    output:
    h2ph_coefficient: height-to-phase conversion factor
    h2ph_coefficient*h: Topographic phase

    """
    # perpendicular baseline[unit:m]
    normal_baseline = np.random.normal(size=(1, 20))*300
    # error of perpendicular baseline
    baseline_erro = np.random.rand(1, 20)
    err_baseline = normal_baseline+baseline_erro
    # compute height-to-phase conversion factor
    h2ph_coefficient = 4*np.pi*err_baseline / \
        (WAVELENGTH*R*np.sin(Incidence_angle))

    return h2ph_coefficient*h, h2ph_coefficient


def sim_phase_noise(noise_level: float) -> np.ndarray:
    """ 
    simulate phase noise based on constant noise level

    input: noise_level

    output: noise

    """
    noise = np.random.normal(loc=0.0, scale=noise_level,
                             size=(1, 20))*(4*np.pi/WAVELENGTH)

    return noise


def search_parm_solution(step: float, Nsearch, A_matrix) -> np.ndarray:
    """
    construct ohase space based on a range of pamrameters we guess:

    step1: creat parameters search space based on  a particular step 、
              search number(range) and orignal parameters 
    step2: caculate param-related phase space such as deformation phase and topographic-height phase

    input: 
    step:  step for search parameters
    2*Nsearch :  number of paramters we can search
    A_matrix :  design matrix concluding temporal or spacial baselines

    output: Search_phase

    """
    parm_space = np.mat(np.arange(-Nsearch*step, Nsearch*step, step))
    phase_space = np.dot(A_matrix, parm_space)

    return phase_space


def model_phase(search_phase1, search_phase2, num_serach) -> np.ndarray:
    """
    compute model_phase based on different paramters (v,h) 
    using kronecker to create a two-dimension matrix to 
    display model_phase(3-dimesion:num_v,num_h,num_ifg)

    input: 
    search_phase1 : v_phase solution space 
    search_phase2 : h_phase solution space
    num_search : the numbers of parameters we search

    output:
    search_pace

    """
    search_space = np.kron(search_phase1, np.ones(
        1, num_serach[1]))+np.kron(np.ones(1, num_serach[0]), search_phase2)

    return search_space


def maximum_temporal_coh(arc_phase, search_space, num_search):
    """caclulate temporal coherence per arc and
       computing (v,h) by maximizing the temporal coherence 
       input: arc_phase: simulated observation phases 
              search_space: model phase
              num_search: number paramters we search
       output: best:best conherence
               best_index :the index of best coherence per (v,h)
               parm_index
    """
    search_size = search_space.shape[1]
    coh_phase = arc_phase*np.ones((1, search_size))-search_space
    # resdual_phase = phase_observation - phase_model
    coh_t = np.sum(np.exp(1j*coh_phase), axis=0)
    # coherence = (1/Nifg)*Σexp(j*resdual_phase)
    best = np.max(coh_t)
    best_index = np.argmax(coh_t, axis=1)
    parm_index = np.unravel_index(
        best_index, (num_search[1], num_search[0]), order="C")

    return best, parm_index, best_index


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
# a = h2phase(40)[0]
# print(a)
# print(h2phase(40)[0].shape)
# simuated_v = 0.1 * WAVELENGTH  # [unit:m/yr]
# simuated_time_range = np.arange(1, 21, 1).reshape(1, 20) * 365 / 12
# print(type(v2phase(simuated_v, simuated_time_range)))
# print(simuated_time_range)
# print(np.array(np.linspace(1, 20, 20) * 0.1 * 4 * np.pi))
# parm = [parm[0]+(a[0]-(Nsearch[0]))*step[0],
#         parm[1]+(a[1]-(Nsearch[1]))*step[1]]

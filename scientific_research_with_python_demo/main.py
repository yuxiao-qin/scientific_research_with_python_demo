import numpy as np
import math

WAVELENGTH = 0.0056  # [m]
H = 780000  # [m]
INCIDENCE_ANGLE = 23 * np.pi / 180  # [radians]
R = H / math.cos(INCIDENCE_ANGLE)  # Slant range [m]

signal_length = 300


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    wrap phase to [-pi,pi]

    input: phase

    output:wrap_phase

    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def get_v2ph_coef(
    temporal_baseline: np.ndarray = np.arange(signal_length) * 12 / 365,
) -> np.ndarray:
    return 4 * np.pi * temporal_baseline / WAVELENGTH


def v2phase(v: float, v2ph_coef: np.ndarray) -> np.ndarray:
    return v * v2ph_coef


def get_h2ph_coef(
    normal_baseline: np.ndarray = np.random.randint(0, 200, size=signal_length)
) -> np.ndarray:
    return 4 * np.pi * normal_baseline / (R * WAVELENGTH * np.sin(INCIDENCE_ANGLE))


def h2phase(h: float, h2phase_coef: np.ndarray) -> np.ndarray:
    return h * h2phase_coef


def noise2phase(noise_level: float) -> np.ndarray:
    return np.random.uniform(0, noise_level, signal_length) * (4 * np.pi / 180)


def generate_phase_noise(noise_level: float, noise_length: int = 30) -> np.ndarray:
    return np.random.normal(loc=0.0, scale=noise_level, size=(1, noise_length))


def sim_arc_phase(
    v: float,
    h: float,
    noise_level: float,
    temporal_baseline: np.ndarray,
    normal_baseline: np.ndarray,
) -> np.ndarray:
    """
    simulate phase of arc between two points based on a module(topographic_height + linear_deformation)

    input:  v: defomation rate per year
            time_range
            h:topographic height per arc
            time_range
            normal_baseline

    output: arc_phase: simulated observation phases = topographic_phase + deformation_phase + nosie

    """
    phase_displacement = v2phase(v, get_v2ph_coef(temporal_baseline))
    phase_height = h2phase(h, get_h2ph_coef(normal_baseline))
    phase_noise = noise2phase(noise_level)

    return wrap_phase(phase_displacement + phase_height + phase_noise)


def search_parm_solution(step: float, Nsearch, A_matrix, param_orig) -> np.ndarray:
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
    parm_space = np.mat(
        np.arange(param_orig - Nsearch * step, param_orig + Nsearch * step, step)
    )
    # parm_space = np.mat(np.arange(0, Nsearch*step, step))
    phase_space = A_matrix @ parm_space

    return phase_space, parm_space


def model_phase(search_phase1, search_phase2, num_serach) -> np.ndarray:
    """
    compute model_phase(φ_model) based on different paramters (v,h) ,
    which is the phase of a number of interferograms phase per arc.
    -----------------------------------------------------------------------------------
    Since we have a range of parameter v and another range of paramters h every iteration,
    we have got phase_height and phase_v whose dimension
    related to its 'number of search solution'.
    In this case , we have to get a combination of phase based on each v and h
    based on 'The multiplication principle of permutations and combinations'

    For example, we get a range of  parameter v (dimension: 1*num_search_v)
    and a range of parameter  h (dimension: 1*num_search_h)
    In one case , we can have a combination of (v,h) (dimension: num_search_v*num_search_h)

    Since we have 'Number of ifg (Nifg)' interferograms, each parmamters will have Nifg phases.
    Then , we get get a range of phase based parameter's pair (v,h)
    named φ_model (dimension: Nifg*(num_search_v*num_search_v)
    ---------------------------------------------------------------------------------
    In our case , we can firtsly computer phase
    based on a range of paramters of Nifg interferograms
    φ_height(dimension:Nifg*num_search_h),
    φ_v(dimension:Nifg*num_search_v).

    Then we have to create a combination φ_height and φ_v in the dimension of interferograms
    φ_model (dimension: Nifg*(num_search_v*num_search_v)
    Kronecker product is introduced in our case,
    we use 'kron' to extend dimension of φ_height or φ_v to
    dimension(Nifg*(num_search_v*num_search_v))
    and then get add φ_model by adding extended φ_height and φ_v.
    ---------------------------------------------------------------------------------
    display model_phase(3-dimesion:num_v,num_h,num_ifg)

    input:
    search_phase1 : v_phase solution space
    search_phase2 : h_phase solution space
    num_search : the numbers of parameters we search

    output:
    search_pace

    """

    search_space = np.kron(search_phase1, np.ones((1, num_serach[1]))) + np.kron(
        np.ones((1, num_serach[0])), search_phase2
    )

    return search_space


def sim_temporal_coh(arc_phase, search_space):
    """caclulate temporal coherence per arc and
    input: arc_phase: simulated observation phases
           search_space: model phase

    output: coh_t : temporal coherence
    """
    search_size = search_space.shape[1]
    coh_phase = arc_phase * np.ones((1, search_size)) - search_space
    # resdual_phase = phase_observation - phase_model
    size_c = coh_phase.shape
    coh_t = np.sum(np.exp(1j * coh_phase), axis=0)
    size_t = coh_t.shape
    # coherence = (1/Nifg)*Σexp(j*resdual_phase)

    return coh_t, size_c, size_t


def maximum_coh(coh_t, num_search):
    best = np.max(coh_t)
    best_index = np.argmax(coh_t)
    best_cot = coh_t[:, best_index]
    param_index = np.unravel_index(
        best_index, (num_search[0], num_search[1]), order="F"
    )

    return best, best_index, param_index, best_cot


def periodogram(v2ph, h2ph, phase_obs, Num_search, step_orig, param_orig):
    v_search = search_parm_solution(step_orig[1], Num_search[1], v2ph, param_orig[1])[0]
    h_search = search_parm_solution(step_orig[0], Num_search[0], h2ph, param_orig[0])[0]
    search_size = [Num_search[1] * 2, Num_search[0] * 2]
    phase_model = model_phase(v_search, h_search, search_size)
    best_coh = sim_temporal_coh(phase_obs, phase_model)
    index = maximum_coh(best_coh[0], search_size)
    param_h = compute_param(index[1], step_orig[0], param_orig[0], search_size[1])
    param_v = compute_param(index[0], step_orig[1], param_orig[1], search_size[0])
    param = [param_v, param_h]
    return param

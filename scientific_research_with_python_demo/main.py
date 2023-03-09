import numpy as np

# Constant
WAVELENGTH = 0.0056  # [unit:m]
H = 780000    # satellite vertical height[m]
Incidence_angle = 23*np.pi/180    # the local incidence angle
R = H/np.cos(Incidence_angle)    # range to the master antenna. test


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    wrap phase to [-pi,pi]

    input: phase

    output:wrap_phase

    """
    return (phase + np.pi) % (2 * np.pi) - np.pi


def sim_arc_phase(v: float, h: float, noise_level: float, time_range, normal_baseline: float) -> np.ndarray:
    """ 
    simulate phase of arc between two points based on a module(topographic_height + linear_deformation)

    input:  v: defomation rate per year
            time_range
            h:topographic height per arc
            time_range
            normal_baseline

    output: arc_phase: simulated observation phases = topographic_phase + deformation_phase + nosie

    """
    v_phase = v2phase(v, time_range)[0]
    h_phase = h2phase(h, normal_baseline)[0]
    v2ph = v2phase(v, time_range)[1]
    h2ph = h2phase(h, normal_baseline)[0]
    noise_phase = sim_phase_noise(noise_level)
    arc_phase = wrap_phase(v_phase + h_phase + noise_phase)

    return arc_phase, v2ph, h2ph


def v_coef(time_baseline) -> np.ndarray:
    """
    caculate v2phase factor

    input: 
        time_baseline:the factor to caclulate temporal baseline

    output: 
    v2phase_coefficeint: v to phase factors
    """
    temporal_step = 12  # [unit:d]
    temporal_samples = temporal_step*time_baseline
    # distance = velocity * days (convert from d to yr because velocity is in m/yr)
    v2phase_coefficeint = 4 * np.pi * temporal_samples / (WAVELENGTH*365)
    return v2phase_coefficeint


def h_coef(normal_baseline: float) -> np.ndarray:
    """
        caculate h2phase factor

    input: 
        normal_baseline:the factor to caclulate temporal baseline

    output: 
    h2phase_coefficeint: h to phase factors
    """
    h2ph_coefficient = 4*np.pi*normal_baseline / \
        (WAVELENGTH*R*np.sin(Incidence_angle))

    return h2ph_coefficient


def coef2phase(coefficeint, param: float) -> np.ndarray:
    """
    Calculate phase difference from velocity difference (of two points).
    and phase difference from topographic height (of two points)

    input:
        param: v or h
        coefficeint: v or h to phase factors
    output:
        phase_model: difference phases based on v ,h or other paramters

    """

    phase_model = coefficeint*param

    return phase_model


# def v2phase(v: float, time_range) -> np.ndarray:
#     """
#     Calculate phase difference from velocity difference (of two points).

#     input:
#     v:defomation rate
#     time_range:the factor to caclulate temporal baseline

#     output:
#     v2phase_coefficeint: temopral baseline
#     v2phase_coefficeint*v: deformation_phase

#     """
#     temporal_baseline = 12  # [unit:d]
#     temporal_samples = temporal_baseline*time_range
#     # distance = velocity * days (convert from d to yr because velocity is in m/yr)
#     v2phase_coefficeint = 4 * np.pi * temporal_samples / (WAVELENGTH*365)

#     return v2phase_coefficeint*v, v2phase_coefficeint  # [unit:rad]


# def h2phase(h: float, normal_baseline: float) -> np.ndarray:
#     """
#     Calculate phase difference from topographic height (of two points)

#     Input:
#         height per acr
#         ormal_baseline : perpendicular baseline[unit:m]

#     output:
#         h2ph_coefficient: height-to-phase conversion factor
#         h2ph_coefficient*h: Topographic phase

#     """

#     # normal_baseline = np.random.normal(size=(1, 20))*300
#     # error of perpendicular baseline
#     # baseline_erro = np.random.rand(1, 20)
#     # err_baseline = normal_baseline+baseline_erro
#     # compute height-to-phase conversion factor
#     h2ph_coefficient = 4*np.pi*normal_baseline / \
#         (WAVELENGTH*R*np.sin(Incidence_angle))

#     return h2ph_coefficient*h, h2ph_coefficient


def sim_phase_noise(noise_level: float) -> np.ndarray:
    """ 
    simulate phase noise based on constant noise level

    input: noise_level

    output: noise

    """
    noise = np.random.uniform(0, noise_level, 20)*(4*np.pi/180)
    # noise = np.random.normal(loc=0.0, scale=noise_level,
    #                          size=(1, 20))*(4*np.pi/180)

    return noise


def param_search(step: float, Nsearch, param_orig):
    """

    input:
        step:  step for search parameters
        2*Nsearch :  number of paramters we can search
    output:
        parm_space: paramters serach space
    """

    parm_space = np.mat(np.arange(param_orig-Nsearch*step,
                        param_orig+Nsearch*step, step))

    return parm_space


def phase_search(coefficeint, param_space) -> np.ndarray:
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
    # parm_space = np.mat(np.arange(param_orig-Nsearch*step,
    #                     param_orig+Nsearch*step, step))
    # parm_space = np.mat(np.arange(0, Nsearch*step, step))
    phase_space = coef2phase(coefficeint, param_space)

    return phase_space


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

    search_space = np.kron(search_phase1, np.ones(
        (1, num_serach[1])))+np.kron(np.ones((1, num_serach[0])), search_phase2)

    return search_space


def sim_temporal_coh(arc_phase, search_space):
    """
       caclulate temporal coherence per arc 
       temporal coherence γ=|(1/Nifgs)Σexp(j*(φ0s_obs-φ0s_modle))|

       input: arc_phase: simulated observation phases 
              search_space: model phase

       output: coh_t : temporal coherence

    """
    search_size = search_space.shape
    coh_phase = arc_phase*np.ones((1, search_size))-search_space
    # resdual_phase = phase_observation - phase_model
    size_c = coh_phase.shape
    coh_t = np.sum(np.exp(1j*coh_phase), axis=0)/20
    size_t = coh_t.shape
    # coherence = γ=|(1/Nifgs)Σexp(j*(φ0s_obs-φ0s_modle))|

    return coh_t


def maximum_coh(coh_t):
    """
    search best coh_t of each paramters (v,h) based on several interferograms
    calculate the chosen (v,h) by converting Linear indexes to subscripts
    here we used a fuction named "np.unravel_index" to get subscripts


    input:
        coh_t: temporal coherence per arc
        num_serach: size of serached paramters
    output:
        best_coh: the max modulus of coh_t
        best_index: Linear indexe of best
        param_index: subscripts of best in the matrix made of searched parameters


    """
    # best_coh = np.max(coh_t)
    best_index = np.argmax(coh_t)
    best_coh = coh_t[:, best_index]

    return best_coh, best_index


def index2sub(best_index, num_search):
    param_index = np.unravel_index(
        best_index, (num_search[0], num_search[1]), order="F")

    return param_index


def compute_param(param_index, step, param_orig, num_search):
    """
    compute paramters by using search_num and subscripts

    input:
        param_index: subscripts of best in the matrix made of searched parameters
        step: step for search parameters
        param_orig: the original param after each iterations
        num_search: size of serached paramters

        output:
        param: (v,h) of max coherence each iterations

    """
    param = param_orig+(param_index+1-num_search)*step
    return param


def periodogram(v2ph, h2ph, phase_obs, Num_search, step_orig, param_orig):
    """
    This is a program named "periodogram" 
    It is an estimator seraching the solution space to find best (v,h),
    based on (topographic_height+linear_deformation)
    which maximize the temporal coherence γ

    input:
        v2ph: velocity-to-phase conversion factor
        h2ph: height-to-phase conversion factor
        phase_obs: simulated obseravation based on 'topographic_height+linear_deformation+niose'
        Num_search: size of solution space
        step_orig: step of searching solution related to parameters
        param_orig: original paramters (v,h)

    output:
        param: The parameters generated after each iteration

    The program consists of a number of modules, 
    which enables users to check and upgrade.

    The modules are:




    """

    v_search = coef2phase(
        v2ph, param_orig[1])
    h_search = coef2phase(
        h2ph,  param_orig[0])
    search_size = [Num_search[1]*2, Num_search[0]*2]
    phase_model = model_phase(v_search, h_search, search_size)
    coh_t = sim_temporal_coh(phase_obs, phase_model)
    best, index = maximum_coh(coh_t)
    sub = index2sub(index)
    param_h = compute_param(
        sub[1], step_orig[0], param_orig[0], search_size[1])
    param_v = compute_param(
        sub[0], step_orig[1], param_orig[1], search_size[0])
    param = [param_v, param_h]
    return param

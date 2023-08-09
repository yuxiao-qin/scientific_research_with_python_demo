import numpy as np


# Constant
WAVELENGTH = 0.056  # [unit:m]
H = 780000  # satellite vertical height[m]
Incidence_angle = 23 * np.pi / 180  # the local incidence angle
R = H / np.cos(Incidence_angle)  # range to the master antenna. test
m2ph = 4 * np.pi / WAVELENGTH


def input_parameters(par2ph, step, Num_search, param_orig, param_name):
    data_set = {
        key: {"par2ph": par2ph[i], "Num_search_max": Num_search[i][0], "Num_search_min": Num_search[i][1], "step_orig": step[i], "param_orig": param_orig[i]}
        for i, key in enumerate(param_name)
    }

    return data_set


def list2dic(param_key: list, param_value: list) -> dict:
    """convert two list to a dictionary

    Parameters
    ----------
    param_key : list
        list of keys
    param_value : list
        list of values

    Returns
    -------
    dict
        dictionary of keys and values
    """
    param_dic = dict(zip(param_key, param_value))

    return param_dic


def compute_Nsearch(std_param: float, step):
    Num_search = round(2 * std_param / step)

    return Num_search


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    """wrap phase to [-pi,pi]

    Parameters
    ----------
    phase : np.ndarray
        unwrapped_phase

    Returns
    -------
     np.ndarray
        phase_wrap:
    """
    phase_wrap = (phase + np.pi) % (2 * np.pi) - np.pi

    return phase_wrap


def unwrap_phase(phase, a_check) -> np.ndarray:
    phase_unwrap = 2 * np.pi * a_check + phase

    return phase_unwrap


def sim_arc_phase(v: float, h: float, v2ph, h2ph: float, SNR) -> np.ndarray:
    """simulate phase of arc between two points based on a module(topographic_height + linear_deformation)

    Parameters
    ----------
    v : float
        defomation rate per year
    h : float
        topographic height per arc
    noise_level : float
        level of uniform noise
    v2ph : _type_
        velocity-to-phase conversion factor
    h2ph : float
        height-to-phase conversion factor

    Returns
    -------
    np.ndarray:
        simulated observation phases = topographic_phase + deformation_phase + nosie
    """
    # v_phase = v2phase(v, time_range)[0]
    # h_phase = h2phase(h, normal_baseline)[0]
    # v2ph = v2phase(v, time_range)[1]
    # h2ph = h2phase(h, normal_baseline)[0]
    v_phase = _coef2phase(v2ph, v)
    h_phase = _coef2phase(h2ph, h)
    phase_unwrap = v_phase + h_phase
    # noise = gauss_noise(phase_unwrap, SNR)
    noise = add_gaussian_noise(phase_unwrap, SNR)
    phase_true = phase_unwrap + noise
    arc_phase = wrap_phase(phase_true)
    # snr_check = check_snr(phase_unwrap, phase_true)
    snr_check = check_snr2(phase_unwrap, noise)
    return arc_phase, snr_check, phase_unwrap


def v_coef(time_baseline) -> np.ndarray:
    """caculate v2phase factor

    Parameters
    ----------
    time_baseline : int
        the factor to caclulate temporal baseline

    Returns
    -------
    np.ndarray
        v to phase factors
    """

    temporal_step = 12  # [unit:d]
    temporal_samples = temporal_step * time_baseline
    # distance = velocity * days (convert from d to yr because velocity is in m/yr)
    v2phase_coefficeint = temporal_samples / 365

    return v2phase_coefficeint


def time_baseline_dt(Nifg, time_range):
    dt = time_range / Nifg
    time = np.arange(1, Nifg + 1, 1).reshape(1, Nifg) * dt / 365

    return time, dt


def h_coef(normal_baseline: float) -> np.ndarray:
    """caculate h2phase factor

    Parameters
    ----------
    normal_baseline : float
        the factor to caclulate temporal baseline

    Returns
    -------
    np.ndarray
        h to phase factors
    """

    # normal_baseline = np.random.normal(size=(1, 20))*300
    # error of perpendicular baseline
    # baseline_erro = np.random.rand(1, 20)
    # err_baseline = normal_baseline+baseline_erro
    h2ph_coefficient = normal_baseline / (R * np.sin(Incidence_angle))

    return h2ph_coefficient


def _coef2phase(coefficeint, param: float) -> np.ndarray:
    """Calculate phase difference from velocity difference (of two points),
    and phase difference from topographic height (of two points).

    Parameters
    ----------
    coefficeint : float
        v or h to phase factors
    param : float
        v or h

    Returns
    -------
    phase_model : np.ndarray
        difference phases based on v ,h or other paramters
    """

    phase_model = m2ph * coefficeint * param

    return phase_model


def sim_phase_noise(noise_level: float, Nifg) -> np.ndarray:
    """simulate phase noise based on constant noise level

    Parameters
    ----------
    noise_level : float
       level of uniform noise

    Returns
    -------
    noise:np.ndarray
        phase noise
    """

    noise = np.random.uniform(0, noise_level, (Nifg, 1)) * (4 * np.pi / 180)
    # noise = np.random.normal(loc=0.0, scale=noise_level,
    #                          size=(1, 20))*(4*np.pi/180)

    return noise


def gauss_noise(signal, noise_level):
    # 给数据加指定SNR的高斯噪声
    noise_lv = np.zeros((signal.size + 1, 1))
    noise_std = np.zeros((signal.size + 1, 1))
    noise_lv[0] = noise_level
    noise_std[0] = np.random.randn(1) * noise_lv[0]
    noise_ph = np.zeros((signal.size, 1))
    for i in range(signal.shape[0]):
        noise_lv[i + 1] = np.random.randn(1) * (np.pi * 5 / 180) + noise_level  # 产生N(0,1)噪声数据
        noise_std[i + 1] = np.multiply(np.random.randn(1), noise_lv[i + 1])
        noise_ph[i] = noise_std[0] + noise_std[i + 1]

    return noise_ph


def add_gaussian_noise(signal, SNR):
    """
    :param signal: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    noise = np.random.randn(*signal.shape)  # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise

    return noise


def check_snr(signal, noise):
    Ps = (np.linalg.norm(signal - signal.mean())) ** 2  # signal power
    Pn = (np.linalg.norm(signal - noise)) ** 2  # noise power
    snr = 10 * np.log10(Ps / Pn)
    return snr


def check_snr2(signal, noise):
    """
    :param signal: 原始信号
    :param noise: 生成的高斯噪声
    :return: 返回两者的信噪比
    """
    signal_power = (1 / signal.shape[0]) * np.sum(np.power(signal, 2))  # 0.5722037
    noise_power = (1 / noise.shape[0]) * np.sum(np.power(noise, 2))  # 0.90688
    SNR = 10 * np.log10(signal_power / noise_power)
    return SNR


def _construct_parameter_space(step: float, Nsearch_max, Nsearch_min, param_orig) -> np.ndarray:
    """create solution searching space

    Parameters
    ----------
    step : float
        step for search parameters
    Nsearch : int
        half number of paramters we can search
    param_orig : float
        original paramters each iterations

    Returns
    -------
    parm_space:np.ndarray
        paramters serach space based on  a particular step 、search number(range) and orignal parameters
    """

    # parm_space = np.mat(np.arange(param_orig-Nsearch*step,
    #                     param_orig+Nsearch*step, step))
    min = np.round(param_orig - Nsearch_min * step, 8)
    max = np.round(param_orig + Nsearch_max * step, 8)
    param_space = np.round(np.linspace(min, max, Nsearch_max + Nsearch_min + 1), 8)

    return param_space


def phase_search(coefficeint, param_space) -> np.ndarray:
    """construct ohase space based on a range of pamrameters we guess:
       step1: creat parameters search space based on  a particular step 、
    #search number(range) and orignal parameters
    # step2: caculate param-related phase space such as deformation phase and topographic-height phase
    Parameters
    ----------
    coefficeint : float
        parameters to phase factors
    param_space : float
         paramter's search space as well as solution serach space
         based on  a particular step 、search number(range) and orignal parameters

    Returns
    -------
    phase_space : np.ndarray
        param-related phase space such as deformation phase and topographic-height phase
    """

    # parm_space = np.mat(np.arange(param_orig-Nsearch*step,
    #                     param_orig+Nsearch*step, step))
    # parm_space = np.mat(np.arange(0, Nsearch*step, step))
    phase_space = _coef2phase(coefficeint, param_space)

    return phase_space


def model_phase(search_phase1, search_phase2, num_serach) -> np.ndarray:
    """compute model_phase(φ_model) based on different paramters (v,h) ,
       which is the phase of a number of interferograms phase per arc.

    Parameters
    ----------
    search_phase1 : float
        v_phase searching space
    search_phase2 : float
        h_phase searching space
    num_serach : _type_
        the numbers of parameters we search[search_size_h,search_size_h]

    Returns
    -------
    search_pace:np.ndarray
        phase searching space based on both v and h

    Notes
    -----
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
    ------------------------------------------------------------------------------------------
    In our case , we can firtsly compute phase
    based on a range of paramters of Nifg interferograms
    φ_height(dimension:Nifg*num_search_h),
    φ_v(dimension:Nifg*num_search_v).

    Then we have to create a combination φ_height and φ_v in the dimension of interferograms
    φ_model (dimension: Nifg*(num_search_v*num_search_v)
    Kronecker product is introduced in our case,
    we use 'kron' to extend dimension of φ_height or φ_v to
    dimension(Nifg*(num_search_v*num_search_v))
    and then get add φ_model by adding extended φ_height and φ_v.
    ------------------------------------------------------------------------------------------

    """

    search_space = np.kron(search_phase1, np.ones((1, num_serach[0]))) + np.kron(np.ones((1, num_serach[1])), search_phase2)

    return search_space


def simulate_temporal_coherence(arc_phase, search_space) -> np.ndarray:
    """caclulate temporal coherence per arc
       temporal coherence γ=|(1/Nifgs)Σexp(j*(φ0s_obs-φ0s_modle))|

    Parameters
    ----------
    arc_phase : float
        simulated observation phases
    search_space : float
        phase searching space based on both v and h

    Returns
    -------
    coh_t:np.ndarray
        temporal coherence
    """

    # size of searched phase_model
    search_size = search_space.shape[1]
    # resdual_phase = phase_observation - phase_model
    coh_phase = arc_phase * np.ones((1, search_size)) - search_space
    Nifg = len(arc_phase)
    coh_t = np.sum(np.exp(1j * coh_phase), axis=0, keepdims=True) / Nifg
    # coherence = γ=|(1/Nifgs)Σexp(j*(φ0s_obs-φ0s_modle))|

    return coh_t


def find_maximum_coherence(coh_t):
    """search best coh_t of each paramters (v,h) based on several interferograms


    Parameters
    ----------
    coh_t : float
        temporal coherence per arc

    Returns
    -------
    best_coh : _type_
        the max modulus of coh_t
    best_index : int
        Linear indexe of best
    """
    # """
    # search best coh_t of each paramters (v,h) based on several interferograms
    # calculate the chosen (v,h) by converting Linear indexes to subscripts
    # here we used a fuction named "np.unravel_index" to get subscripts

    # input:
    #     coh_t: temporal coherence per arc
    #     num_serach: size of serached paramters
    # output:
    #     best_coh: the max modulus of coh_t
    #     best_index: Linear indexe of best
    #     param_index: subscripts of best in the matrix made of searched parameters

    # """
    # best_coh = np.max(coh_t)
    best_index = np.argmax(abs(coh_t))
    best_coh = coh_t[:, best_index]

    return best_coh, best_index


def index2sub(best_index, num_search):
    """converting Linear indexes to subscripts
       here we used a fuction named "np.unravel_index" to get subscripts

    Parameters
    ----------
    best_index : int
        Linear indexe of best
    num_search : int
        size of serached paramters h and v

    Returns
    -------
    param_index : int
        subscripts of best in the matrix made of searched parameters
    """

    param_index = np.unravel_index(best_index, (num_search[0], num_search[1]), order="F")

    return param_index


def compute_param(param_index, step, param_orig, num_search):
    """compute paramters by using search_num and subscripts

    Parameters
    ----------
    param_index : int
        subscripts of best in the matrix made of searched parameters
    step : float
        step for search parameters
    param_orig : float
        the original param after each iterations
    num_search : int
        size of serached paramters

    Returns
    -------
    param: float
        (v,h) of max coherence each iterations
    """

    param = np.round(param_orig + (param_index - num_search) * step, 8)

    return param


def correct_h2ph(h2ph, n):
    """caculate factor to correct h

    Parameters
    ----------
    h2ph : float
        h to phase factor per arc
    n : int
        the index of  choosen arc

    Returns
    -------
    correct_factor : float
        correcting factor as of result of using  mean_h2ph to estimate parameters
    """
    mean_h2ph = np.mean(h2ph, axis=1)
    factors = h2ph[:, n] / mean_h2ph
    correct_factor = np.median(factors)

    return correct_factor


def resedual_phase(h2ph, v2ph, param, best, phase_obs):
    """compute phase resudual
    phase_model = phase_obs + 2pi*a_check + phase_resedual

    Parameters
    ----------
    v2ph : float
        v to phase factor
    h2ph : float
        h to phase factor
    param : float
        estimated paramters based on periodogram
    best : float
        max temporal coherence we searched (complex number)
    phase_obs : float
        observation phase

    Returns
    -------
    phase_resedual : float
        wrapped phase_resedual based on phase_model and phase_obs after parameters estimation
    phase_model : float
        phase based on parameters(v,h) we estimate
    """
    phase_model = _coef2phase(h2ph, param[0]) + _coef2phase(v2ph, param[1]) + np.angle(best)
    phase_ambiguity = phase_obs - phase_model
    phase_resedual = wrap_phase(phase_ambiguity)

    return phase_resedual, phase_model


def compute_ambiguity(observed_phase, modeled_phase, residual_phase):
    """compute interger ambiguity based on  round

    Parameters
    ----------
    phase_obs : float
        obseravation phase
    phase_model : float
        phase_model based on parameters (v,h) we estimate
    phase_resedual : float
        wrapped phase_resedual based on phase_model and phase_obs after parameters estimation

    Returns
    -------
    a_check : int
        phase ambiguity
    """
    estimated_ambiguities = np.round((modeled_phase + residual_phase - observed_phase) / (2 * np.pi))

    return estimated_ambiguities


def model_matrix(v2ph, h2ph):
    design_matrix = np.hstack((h2ph, v2ph))

    return design_matrix


def correct_param(A_design, phase_unwrap):
    """correct parameter using unwrap_phase based on leaast-square estimation

    Parameters
    ----------
    A_design : float
        design matrix consits of h2ph and v2ph
    phase_unwrap : float
        unwrapped phase based on parameters we estimated


    unwrapped_phase : variable
    unwrap_phase: func

    Returns
    -------
    param_correct : float
        the new parameters caculated by usuing least-square estimator
    """
    N = np.mat(np.dot(A_design.T, A_design))
    R = np.mat(np.linalg.cholesky(N)).T
    rhs = (R.I) * (((R.T).I) * A_design.T)
    param_correct = rhs * phase_unwrap

    return param_correct


def ambiguity_solution(data_set, n, best, phase_obs):
    # ---------------------------------------
    # Correct etsimated dH's for mean_h2ph
    # ---------------------------------------
    factors = correct_h2ph(data_set["height"]["par2ph"], n)
    param1 = data_set["height"]["param_orig"] / factors
    param2 = data_set["velocity"]["param_orig"]
    # ---------------------------------------
    # caculate phase resedual of ambiguities
    # ---------------------------------------
    phase_resedual, phase_model = resedual_phase(data_set["height"]["par2ph"], data_set["velocity"]["par2ph"], [param1, param2], best, phase_obs)

    # ---------------------------------------
    # caculate ambiguities
    # ---------------------------------------
    a_check = compute_ambiguity(phase_obs, phase_model, phase_resedual)

    return a_check


def data_prepare(params):
    if params["est_flag"] == 0:  # normal case
        return params
    elif params["est_flag"] == 1:  # test of v
        params["v_orig"] = np.linspace(params["v_range"][0], params["v_range"][1], 50)
    elif params["est_flag"] == 2:  # test of h
        params["h_orig"] = np.linspace(params["h_range"][0], params["h_range"][1], 50)
    elif params["est_flag"] == 3:  # test of Nifg
        params["Nifg_orig"] = np.arange(params["Nifg_range"][0], params["Nifg_range"][1], 1)
    elif params["est_flag"] == 4:  # test of std_v
        params["std_v"] = np.linspace(params["std_v_range"][0], params["std_v_range"][1], 100)
    elif params["est_flag"] == 5:  # test of std_h
        params["std_h"] = np.linspace(params["std_h_range"][0], params["std_h_range"][1], 100)

    return params


def design_mat(h2ph, v2ph, phase_obs, pseudo_param):
    Nifg = len(phase_obs)
    par2ph = np.hstack((h2ph, v2ph)) * m2ph
    a_mat = 2 * np.pi * np.eye(Nifg)
    A_1 = np.hstack((a_mat, par2ph))
    P = np.hstack((np.zeros((2, Nifg)), np.eye(2)))
    A_design = np.vstack((A_1, P))
    y = np.vstack((phase_obs, pseudo_param))

    return A_design, y


def cov_obs(sig2, std_param):
    Q_y1 = np.diag(sig2)
    Q_y2 = np.diag(std_param)
    Q_0 = np.zeros((len(sig2), len(std_param)))
    Q_up = np.hstack((Q_y1, Q_0))
    Q_down = np.hstack((Q_0.T, Q_y2))
    Q_y = np.vstack((Q_up, Q_down))
    return Q_y


def cov_ahat(C, Q_y, Nifg):
    C_1 = np.linalg.inv(C)
    Q_chat = np.dot(C_1, np.dot(Q_y, C_1.T))
    # 取Q_chat n行到n列的元素
    Q_ahat = Q_chat[0:Nifg, 0:Nifg]
    return Q_ahat


def compute_ahat(A_design, y):
    a_hat = np.dot(np.linalg.inv(np.dot(A_design.T, A_design)), np.dot(A_design.T, y))

    return a_hat

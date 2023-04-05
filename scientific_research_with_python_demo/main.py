import utils


def periodogram(v2ph, h2ph, phase_obs, Num_search, step_orig: float, param_orig):
    """This is a program named "periodogram"
       It is an estimator seraching the solution space to find best (v,h),
       based on (topographic_height+linear_deformation)
       which maximize the temporal coherence γ

    Parameters
    ----------
    v2ph : float
       velocity-to-phase conversion factor
    h2ph : _type_
        height-to-phase conversion factor
    phase_obs : _type_
        simulated obseravation based on 'topographic_height+linear_deformation+niose'
    Num_search : _type_
        size of solution space
    step_orig : dict
        step of searching solution related to parameters[step_h,step_v]
    param_orig : _type_
        original paramters (h,v)

    Returns
    -------
    param : _type_
        The parameters generated after each iteration

    Notes
    -----
    The program consists of a number of modules,
    which enables users to check and upgrade.

    The modules are:
        param_serach:
           creat solution searching space
        coef2phase:
           compute phase based on baselines
        model_phase:
           computer
    """

    # ---------------------------------------------------------
    #  Step 1: Do something...
    # ---------------------------------------------------------
    search = dict()  # TODO: HOw to we initialize a dict?
    phase = dict()
    for key in ("height", "velocity"):
        search[key] = utils._construct_parameter_space(step_orig[key], Num_search[key], param_orig[key])
        par2ph = v2ph if key == "velocity" else h2ph
        phase[key] = utils._coef2phase(par2ph, search[key])

    # search_size=[serach_sizeH,serach_sizeH]
    search_size = [Num_search[0] * 2, Num_search[1] * 2]

    # kronecker积
    phase_model = utils.model_phase(phase["velocity"], phase["height"], search_size)

    coh_t = utils.simulate_temporal_coherence(phase_obs, phase_model)
    best, index = utils.find_maximum_coherence(coh_t)
    sub = utils.index2sub(index, search_size)

    param_h = utils.compute_param(sub[0], step_orig[0], param_orig[0], Num_search[0])
    param_v = utils.compute_param(sub[1], step_orig[1], param_orig[1], Num_search[1])

    param = [param_h, param_v]

    return param

import numpy as np

WAVELENGTH = 0.0056  # [unit:m]
def wrap_phase(phase:np.ndarray)->np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi# 缠绕相位,测试，测试,test

def construct_simulated_arc_phase(dv:float, dh:float, noise_level:float)->np.ndarray:

    dv_phase = v2phase(dv)
    dh_phase = h2phase(dh)
    noise_phase = generate_phase_noise(noise_level)*4*np.pi/WAVELENGTH
    arc_phase=dv_phase + dh_phase + noise_phase
    return arc_phase

def v2phase(v:float, time_range:np.ndarray=np.arange(20))->np.ndarray:
    """Calculate phase difference from velocity difference (of two points).
    """
    temporal_baseline = 12  # [unit:d]
    Btemp=temporal_baseline*time_range
    # distance = velocity * days (convert from d to yr because velocity is in m/yr)
    distance = v * Btemp / 365   # [unit:m]
    return distance * 4 * np.pi / WAVELENGTH  # [unit:rad]

def h2phase(h:float)->np.ndarray:
    B_normal=300#[unit:m]
    H=780000# satellite vertical height[m]
    theta=23*np.pi/180
    R=H/np.cos(theta)
    Bn=4*np.pi*B_normal/(WAVELENGTH*R*np.sin(theta))
    return Bn*h*4*np.pi/WAVELENGTH

def generate_phase_noise(noise_level:float)->np.ndarray:
    noise=np.random.normal(loc=0.0,scale=noise_level,size=(1,20))
    return noise

def construct_param_search_space(step:float,Nsearch,A_matrix)->np.ndarray:
    """
    construct estimated parameters' search space
    such as height [-10:1:10]
    input: step  :step for search
           2*Nsearch :number of paramters we can search   
    创建参数搜索区域[-10:1:10]"""
    Search_space=np.mat(np.arange(-Nsearch*step,Nsearch*step,step))
    Search_space_phase=np.dot(A_matrix,Search_space)
    return Search_space_phase

def construct_search_space(phase_search1,phase_serach2,num_serach)->np.ndarray:
    search_space=np.kron(phase_search1,np.ones(1,num_serach[1]))+np.kron(np.ones(1,num_serach[0]),phase_serach2) 
    return search_space

def maximum_main(dphase,search_space,row_num_serach,num_search,parm,Nsearch):
    coh_phase=dphase*np.ones((1,row_num_serach))-search_space
    coh_t=np.sum(np.exp(1j*coh_phase),axis= 0)
    best=np.max(coh_t)
    best_index=np.argmax(coh_t,axis=1)
    #caculate parameters
    a=np.unravel_index(best_index,(num_search[1],num_search[0]),order="C")
    parm=[parm[0]+(a[0]-(Nsearch[0]+1)),parm[1]+(a[1]-(Nsearch[1]+1))]
    return best,parm



v_orig=0.01
h_orig=[10]*20+np.random.normal(loc=0,scale=1,size=(1,20))
noise_level=0.1
phase_unwrapped=construct_simulated_arc_phase(v_orig,h_orig,noise_level).T
print(h_orig)
print(h2phase(h_orig))
print(wrap_phase(phase_unwrapped))
# print(type(generate_phase_noise(0.1)))
# def estimate_parameters(constructed_simulated_phase):
#     # TODO: implement this function
#     return est_dv, est_dh
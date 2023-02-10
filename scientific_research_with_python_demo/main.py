import numpy as np

WAVELENGTH = 0.0056  # [unit:m]
def wrap_phase(phase:np.ndarray)->np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi# 缠绕相位,测试

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
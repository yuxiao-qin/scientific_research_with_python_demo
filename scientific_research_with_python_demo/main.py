import numpy as np

WAVELENGTH = 0.0056  # [unit:m]
R0 = 786000 # [unit:m]
INCIDENCE_ANGLE = 39.14 # [unit:deg]


def wrap_phase(phase:np.ndarray)->np.ndarray:
    return (phase + np.pi) % (2 * np.pi) - np.pi

def construct_simulated_arc_phase(dv:float, dh:float, noise_level:float)->np.ndarray:

    dv_phase = v2phase(dv)
    dh_phase = h2phase(dh)
    noise_phase = generate_phase_noise(noise_level)

    return wrap_phase(dv_phase + dh_phase + noise_phase)

def v2phase(v:float, time_range:np.ndarray=np.arange(30))->np.ndarray:
    """Calculate phase difference from velocity difference (of two points).
    """
    temporal_baseline = 12  # [unit:d]
    # distance = velocity * days (convert from d to yr because velocity is in m/yr)
    distance = v * time_range * temporal_baseline / 365   # [unit:m]
    return distance * 4 * np.pi / WAVELENGTH  # [unit:rad]

def h2phase(h:float, normal_baseline_range:np.ndarray=np.arange(0, 600, 20))->np.ndarray:
    """Calculate phase difference from height difference (of two points).
    """
    distance = h * normal_baseline_range / (R0 * np.sin(INCIDENCE_ANGLE * np.pi / 180)) # [unit:m]
    return distance * 4 * np.pi / WAVELENGTH  # [unit:rad]

def generate_phase_noise(noise_level:float, noise_length:int = 30)->np.ndarray:
    noise = np.random.normal(loc=0.0, scale=noise_level, size=(1, noise_length))
    return noise







# def estimate_parameters(constructed_simulated_phase):
#     # TODO: implement this function
#     return est_dv, est_dh
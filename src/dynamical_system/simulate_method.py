from os.path import dirname, abspath
import sys
import numpy as np
from tqdm import tqdm
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
### Import module
from dynamical_system.function import runge_kutta_4

def simulate_rk4(system_model, 
                model_params, 
                initial_state, 
                simulation_settings={'number of samples':None,
                                    'start time':0.,
                                    'ending time':50.,
                                    'time step size':1e-2,},
                dynamical_noise_settings={'dynamical noise': False,
                                        'covariance matrix': None},
                random_seed=0,):
    
    """Simulate continuous system using 4th order Runge-Kutta(RK4).
    
    Args:
        system_model (def:function): Contunious system's function (ODE: Ordinaly Differential Equation)
        model_params (list): Parameters of the function
        X0 (numpy array): Initial state vector (n, ) or (n, 1)
        initial_state (bool): process_noise
        simulation_settings:
        dynamical_noise_settings:
        random_seed:
    
    Returns:
        X (numpy array): Result of simulation (N, n)
        t (numpy array): time data (N)
    """
    
    ### Simulation settings
    f = system_model
    f_params = model_params
    X0 = initial_state
    N = simulation_settings['number of samples']
    t_stt = simulation_settings['start time']
    t_end = simulation_settings['ending time']
    dt = simulation_settings['time step size']
    t = np.arange(t_stt, t_end+dt, dt)
    h = dt
    d = len(X0)
    if N==None:
        N = t.shape[0]
    X = np.zeros((N, d))
    X[0] = X0
    
    ### Dynamical noise
    dynamical_noise = dynamical_noise_settings['dynamical noise']
    covariance_dynamical_noise = dynamical_noise_settings['covariance matrix']
    if dynamical_noise:
        np.random.seed(seed=random_seed)
        v = np.random.multivariate_normal(mean=np.zeros(d), cov=covariance_dynamical_noise, size=N)
    else:
        v = np.zeros(((N, d)))

    ### Simulation
    for k in tqdm(range(N-1), desc="Simulation", leave=False):
        X[k+1] = runge_kutta_4(t[k], X[k], f, f_params, h) + v[k]

    return X, t[:X.shape[0]]
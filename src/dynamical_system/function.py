import numpy as np

### 4 order Runge-kutta 
def runge_kutta_4(t, X, f, f_params, h):
    k1 = h * f(t, X, f_params)
    k2 = h * f(t + h / 2, X + k1 / 2, f_params)
    k3 = h * f(t + h / 2, X + k2 / 2, f_params)
    k4 = h * f(t + h, X + k3, f_params)
    k = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) 
    return X + k
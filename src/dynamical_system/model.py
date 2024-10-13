import numpy as np

### Lorenz
def lorenz(t, X, parameter_list):
    dX = np.zeros(X.shape)
    [sigma, rho, beta] = parameter_list
    dX[0] = sigma * (X[1] - X[0])
    dX[1] = X[0] * (rho - X[2]) - X[1]
    dX[2] = X[0] * X[1] - beta * X[2]
    return dX

### Rossler
def rossler(t, X, parameter_list):
    dX = np.zeros(X.shape)
    [a, b, c] = parameter_list
    dX[0] = -(X[1] + X[2])
    dX[1] = X[0] + a * X[1]
    dX[2] = b + X[2] * (X[0] - c)
    return dX
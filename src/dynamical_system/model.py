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

### Langford
def langford(t, X, parameter_list):
    dX = np.zeros(X.shape)
    [a, b, c, d, e, f] = parameter_list
    #dX[0] = a * X[0] + b * X[1] + X[0] * X[2]
    #dX[1] = c * X[0] + d * X[1] + X[1] * X[2]
    #dX[2] = e * X[2] - (X[0] * X[0] + X[1] * X[1] + X[2] * X[2])
    dX[0] = (X[2] - b) * X[0] - d * X[1]
    dX[1] = d * X[0] + (X[2] - b) * X[1]
    dX[2] = c + a * X[2] - X[2] * X[2] * X[2] / 3 - (X[0] * X[0] + X[1] * X[1]) * (1 + e * X[2]) + f * X[2] * X[0] * X[0] * X[0]
    return dX

def torus(t, X, parameter_list):
    dX = np.zeros(X.shape)
    [a] = parameter_list
    dX[0] = X[1]
    dX[1] = -X[0] - X[1] * X[2]
    dX[2] = X[1] * X[1] - a
    return dX
import numpy as np
from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')

def data_discretization(data, bins=200):
        _, bins = np.histogram(data, bins, density=True)
        bins_indices = np.digitize(data, bins)
        data_discrete = data[bins_indices]
        return data_discrete

def mutual_information(X, Y, bins=10):
    p_xy, xedges, yedges = np.histogram2d(X, Y, bins=bins, density=True)
    p_x, _ = np.histogram(X, bins=xedges, density=True)
    p_y, _ = np.histogram(Y, bins=yedges, density=True)
    p_x_p_y = p_x[:, np.newaxis] * p_y
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    i = np.sum(p_xy * np.ma.log(p_xy / p_x_p_y) * dx * dy)
    return i

class FindTimeDelay():

    def __init__(self, data, search_size, method='mutual_information'):
        self.X = data
        self.T = search_size
        self.method = method

    def run(self, bins=10, return_I=False):
        self.I = np.zeros(self.T)
        for i in tqdm(range(0, self.T), desc='Calculating['+self.method+']', leave=False):
            X_unlagged = self.X[:-i]
            X_lagged = np.roll(self.X, -i)[:-i]
            if self.method=='mutual_information': self.I[i] = mutual_information(X_unlagged, X_lagged, bins)
        if return_I: return self.I

    def result(self):
        tau = None
        for i in range(0, self.T):
            if tau is None and i > 1 and self.I[i - 1] < self.I[i]:
                tau = i - 1 # first　minimum time-delay
        if tau is None:
            tau = 0 # cannot find time-delay
            print('Cannot find time-delay!')
        return  tau
        
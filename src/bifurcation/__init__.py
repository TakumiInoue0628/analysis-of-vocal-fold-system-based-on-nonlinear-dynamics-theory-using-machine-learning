from os.path import dirname, abspath
import sys
import itertools
import numpy as np
from scipy.signal import savgol_filter
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from bifurcation.function import *

def local_maxima(data_list,
                 parameter_list, 
                 prominence=1.,
                 height=None,
                 distance=None,
                 threshold=None,
                 savgol_filtering=False,
                 return_flatten=False,
                 savgol_filter_params={'window length': 1,
                                        'polyorder': 1}
                ):
    
    """Get the local maxima using "find peaks".
    
    Args:
        data_list (list): 
        parameter_list (list): 
        prominence (int):
        return_flatten (bool):
    
    Returns:
        localmaxima_list (list): 
        localmaxima_idx_list (list): 
        localmaxima_parameter_list (list): 
    """

    localmaxima_list = []
    localmaxima_idx_list = []
    localmaxima_parameter_list = []
    for i in range(len(data_list)):
        if savgol_filtering: data = savgol_filter(data_list[i], window_length=savgol_filter_params['window length'], polyorder=savgol_filter_params['polyorder'])
        else: data = data_list[i]
        peaks, idx = fing_peaks_index(data, prominence, height, distance, threshold)
        localmaxima_list.append(peaks)
        localmaxima_idx_list.append(idx)
        localmaxima_parameter_list.append(np.full(len(peaks), parameter_list[i][0]))

    if return_flatten:
        localmaxima_flatten = list(itertools.chain.from_iterable(localmaxima_list))
        localmaxima_idx_flatten = list(itertools.chain.from_iterable(localmaxima_idx_list))
        localmaxima_parameter_flatten = list(itertools.chain.from_iterable(localmaxima_parameter_list))
        return localmaxima_list, localmaxima_idx_list, localmaxima_parameter_list, localmaxima_flatten, localmaxima_idx_flatten, localmaxima_parameter_flatten
    else:
        return localmaxima_list, localmaxima_idx_list, localmaxima_parameter_list
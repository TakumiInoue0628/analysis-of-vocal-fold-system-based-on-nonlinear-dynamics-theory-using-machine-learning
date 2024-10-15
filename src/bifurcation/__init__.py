from os.path import dirname, abspath
import sys
import itertools
import numpy as np
from scipy.signal import savgol_filter
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from bifurcation.function import *
from util import Normalization

def local_maxima(data_list,
                 parameter_list, 
                 standard_scaler=False,
                 savgol_filtering=False,
                 return_flatten=False,
                 find_peaks_params={'prominence':1, 
                                    'height':None, 
                                    'distance':None, 
                                    'threshold':None},
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

        if standard_scaler:
            data = (data_list[i] - np.mean(data_list[i])) / np.std(data_list[i]) 
        else: data = data_list[i]

        if savgol_filtering: data = savgol_filter(data, window_length=savgol_filter_params['window length'], polyorder=savgol_filter_params['polyorder'])

        peaks, idx = fing_peaks_index(data, prominence=find_peaks_params['prominence'], height=find_peaks_params['height'], distance=find_peaks_params['distance'], threshold=find_peaks_params['threshold'])
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
import os
import random
import numpy as np
import pandas as pd
import cv2
from scipy.signal import buttord, butter, filtfilt
from scipy.ndimage import convolve
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import tqdm

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def load_csv_data(csv_path, data_name_list, sample_span, quiet=False):
    if quiet==False:
        print('Loading csv data')
        print('file path | '+csv_path)
        print('data list | '+", ".join(data_name_list))
    elif quiet==True:
        pass
    data_df = pd.read_csv(csv_path)
    data_list = []
    for i in range(len(data_name_list)):
        data_list.append(data_df[[data_name_list[i]]].values[sample_span[0]:sample_span[1], 0])
    index = np.arange(sample_span[0], sample_span[1])
    return data_list, index

def load_video_data(video_path, time_span, shooting_time_interval=1/10000, to_GRAY=True,):
    print('Loading video data')
    print('file path | '+video_path)
    ### VideoCapture (get object)
    cap = cv2.VideoCapture(video_path)
    ### get video property
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ### generate video time data
    video_t = np.arange(0, frame_count+1)*shooting_time_interval
    ### decide start load point & end load point
    start_frame = np.argmin(abs(video_t-time_span[0]))
    stop_frame = np.argmin(abs(video_t-time_span[1]))
    t_data = video_t[start_frame:stop_frame] 
    ### load video
    frames = []
    for i in tqdm.tqdm(range(stop_frame), desc="Loading", leave=False):
        ret, frame = cap.read()
        if ret: # read successed
            if i > int(start_frame-1):
                ### RGB 3ch --> GRAY 1ch
                if to_GRAY:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ### save frame
                frames.append(frame)
        else : # read failed
            break
    cap.release()
    ### convert datatype: list init8 --> numpy float64
    video_data = np.array(frames).astype(np.float64)
    return video_data, t_data

def bandpass_filter(data, t_data, passband_edge_freq, stopband_edge_freq, passband_edge_max_loss, stopband_edge_min_loss,):
    dt = t_data[1] - t_data[0]
    sampling_rate = 1. / dt
    niquist_freq = sampling_rate / 2.
    passband_edge_freq_normalize = passband_edge_freq / niquist_freq
    stopband_edge_freq_normalize = stopband_edge_freq / niquist_freq
    butterworth_order, butterworth_natural_freq = buttord(
                                                        wp=passband_edge_freq_normalize, 
                                                        ws=stopband_edge_freq_normalize,
                                                        gpass=passband_edge_max_loss,
                                                        gstop=stopband_edge_min_loss
                                                        )
    numerator_filterfunc, denominator_filterfunc = butter(
                                                        N=butterworth_order,
                                                        Wn=butterworth_natural_freq,
                                                        btype='band'
                                                        )
    data_filtered = filtfilt(
                            b=numerator_filterfunc,
                            a=denominator_filterfunc,
                            x=data
                            )
    return data_filtered

def video_filtering(video_data, kernel_length, kernel_size):
    k = np.ones((kernel_length, kernel_size, kernel_size))/float(kernel_size*kernel_size*kernel_length)
    video_data_filtered = convolve(video_data, k)
    return video_data_filtered

def video_savgol_filtering(video_data, window_length, polyorder):
    video_data_filtered = savgol_filter(video_data, window_length, polyorder, axis=0)
    return video_data_filtered

def line_scanning(video_data, position, width):
    kymograph_data = video_data[:, position[1], (position[0]-int(width/2)):(position[0]+int(width/2))]
    return kymograph_data

def standardize(data):
    standardscaler = StandardScaler()
    if data.ndim==1:
        data = data.reshape(-1, 1)
        standardscaler.fit(data)
        return (standardscaler.transform(data)).squeeze()
    else:
        standardscaler.fit(data)
        return standardscaler.transform(data)
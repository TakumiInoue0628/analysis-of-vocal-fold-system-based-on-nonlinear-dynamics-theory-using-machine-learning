import numpy as np

def fft(data, t_data):
    dt = t_data[1] - t_data[0]
    f = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[0], dt)
    amp = np.abs(f/(data.shape[0]/2))
    plt_lim = int(data.shape[0]/2)
    return freq[1:plt_lim], amp[1:plt_lim]

def short_time_ft(data, dt, nperseg):
    from scipy.signal import stft
    return stft(x=data, fs=1/dt, nperseg=nperseg)

def bandpass_filter(data, time, params):
    from scipy.signal import buttord, butter, filtfilt
    passband_edge_freq = params['passband_edge_freq']
    stopband_edge_freq = params['stopband_edge_freq']
    passband_edge_max_loss = params['passband_edge_max_loss']
    stopband_edge_min_loss = params['stopband_edge_min_loss']
    dt = time[1] - time[0]
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

class Normalization():

    def __init__(self, zero_mean=True, unit_standard_deviation=True):
        self.zero_mean = zero_mean
        self.one_std = unit_standard_deviation

    def transform(self, X):
        X_t = X
        if self.zero_mean:
            self.X_mean = X.mean(axis=0)
            X_t -= self.X_mean
        if self.one_std:
            self.X_std = X.std(axis=0)
            X_t /= self.X_std
        return X_t

    def inverse_transform(self, X_t):
        X = X_t
        if self.one_std:
            X *= self.X_std
        if self.zero_mean:
            X += self.X_mean
        return X
    
def find_steady_range(data, window_length, threshold, margin):
        diff = np.abs(np.diff(data[::window_length]))
        idxs = np.where(diff<threshold)[0] * window_length
        return (idxs[0]+margin, idxs[-1]+window_length-margin)
    
def timedelay_embedding(x, embedding_dimension, delay):
    N = len(x)
    m = embedding_dimension
    n = N - (m - 1) * delay
    X_embed = np.zeros((n, m))
    for i in range(m): X_embed[:, i] = x[(i * delay):(i * delay + n)]
    return X_embed

### Morphological operations (image processing)

def erosion(image, boundary='edge'):
  pad_image = np.pad(image, 1, boundary)
  areas = np.lib.stride_tricks.as_strided(pad_image, image.shape + (3, 3), pad_image.strides * 2)
  return np.min(areas, axis=(2, 3))

def dilation(image, boundary='edge'):
  pad_image = np.pad(image, 1, boundary)
  areas = np.lib.stride_tricks.as_strided(pad_image, image.shape + (3, 3), pad_image.strides * 2)
  return np.max(areas, axis=(2, 3))

def morophological_operations(image, process_type='opening', n_erosion=1, n_dilation=1, boundary='edge'):
    if process_type == 'opening':
        for _ in range(n_erosion): image = erosion(image, boundary)
        for _ in range(n_dilation): image = dilation(image, boundary)
    if process_type == 'closing':
        for _ in range(n_dilation): image = dilation(image, boundary)
        for _ in range(n_erosion): image = erosion(image, boundary)
    return image

def gray_to_rgb(image_gray):
    image_rgb = (np.stack((image_gray,)*3, axis=-1) - image_gray.min()) / (image_gray.max() - image_gray.min())
    return (image_rgb * 255).astype('uint8')
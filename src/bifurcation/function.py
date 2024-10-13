from scipy.signal import find_peaks

def fing_peaks_index(data, prominence, height=None, distance=None, threshold=None):
    peaks_index, _ = find_peaks(data, prominence=prominence, height=height, distance=distance, threshold=threshold)
    peaks = data[peaks_index]
    return peaks, peaks_index
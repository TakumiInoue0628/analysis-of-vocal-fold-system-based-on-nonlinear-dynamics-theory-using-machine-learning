from os.path import dirname, abspath
import sys
import numpy as np
from scipy.signal import savgol_filter
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from data_loading.function import *

def LoadCSV(file_path, data_name_list, sample_span):
    data_list, _ = load_csv_data(file_path, data_name_list, sample_span)
    return data_list

def LoadCSVandVIDEOS(csv_path, data_name_list, sample_span, 
                      videos_path_list, videos_shooting_time_interval=1/10000, uppercase_t=False):
    csv_data_list, _ = load_csv_data(csv_path, data_name_list, sample_span)
    ### generate new csv time data　(to match the time data of csv and video)
    if uppercase_t: csv_t_data_list, _ = load_csv_data(csv_path, ['T'], sample_span, quiet=True)
    else: csv_t_data_list, _ = load_csv_data(csv_path, ['t'], sample_span, quiet=True)
    csv_t_data = csv_t_data_list[0]
    ### LOAD VIDEO (same csv span)
    video_data_list = []
    video_t_data_list = []
    for i in range(len(videos_path_list)):
        video_data, video_t_data = load_video_data(
                                            video_path=videos_path_list[i],
                                            time_span=[csv_t_data[0], csv_t_data[-1]],
                                            shooting_time_interval=videos_shooting_time_interval
                                            )
        video_data_list.append(video_data)
        video_t_data_list.append(video_t_data)
    return csv_data_list, video_data_list, video_t_data_list

def LoadCSVandKYMOGRAMS(csv_path, data_name_list, sample_span, 
                      kymograms_path_list, kymograms_time_path_list, uppercase_t=False):
    csv_data_list, _ = load_csv_data(csv_path, data_name_list, sample_span)
    ### generate new csv time data　(to match the time data of csv and video)
    if uppercase_t: csv_t_data_list, _ = load_csv_data(csv_path, ['T'], sample_span, quiet=True)
    else: csv_t_data_list, _ = load_csv_data(csv_path, ['t'], sample_span, quiet=True)
    csv_t_data = csv_t_data_list[0]
    ### LOAD VIDEO (same csv span)
    kymogram_data_list = []
    kymogram_time_data_list = []
    for i in range(len(kymograms_path_list)):
        kymogram_time_data = np.loadtxt(kymograms_time_path_list[i])
        stt_idx = np.argmin(abs(kymogram_time_data-csv_t_data[0]))
        end_idx = np.argmin(abs(kymogram_time_data-csv_t_data[-1]))
        kymogram_data = np.loadtxt(kymograms_path_list[i])
        kymogram_data_list.append(kymogram_data[stt_idx:end_idx+1])
        kymogram_time_data_list.append(kymogram_time_data[stt_idx:end_idx+1])
    return csv_data_list, kymogram_data_list, kymogram_time_data_list

class PreProcessing():

    def __init__(self, data, t_data, video_data_list, video_t_data_list):
        self.raw_data = data
        self.data = data
        self.t_data = t_data
        self.video_data_list = video_data_list
        self.video_t_data_list = video_t_data_list

    def cut(self, sample_span, new_t=False):
        self.data = self.data[sample_span[0]:sample_span[1]]
        self.t_data = self.t_data[sample_span[0]:sample_span[1]]
        video_data_list_cut = []
        video_t_data_list_cut = []
        for i in range(len(self.video_data_list)):
            idx_start = np.abs(np.asarray(self.video_t_data_list[i])-self.t_data[0]).argmin()
            idx_stop = np.abs(np.asarray(self.video_t_data_list[i])-self.t_data[-1]).argmin()
            video_data_list_cut.append(self.video_data_list[i][idx_start:idx_stop])
            video_t_data_list_cut.append(self.video_t_data_list[i][idx_start:idx_stop])
        if new_t: ### t0=0, t1=0+dt, t2=0+2dt ...
            self.t_data = np.arange(0, self.t_data.shape[0])*(self.t_data[1]-self.t_data[0])
            video_t_data_list_cut = []
            for i in range(len(self.video_data_list)):
                video_t_data_list_cut.append(np.arange(0, self.video_t_data_list[i].shape[0])*(self.video_t_data_list[i][1]-self.video_t_data_list[i][0]))
        self.video_data_list = video_data_list_cut
        self.video_t_data_list = video_t_data_list_cut

    def filter(self, method='bandpass_filtering', params={'passband_edge_freq':np.array([90, 200]), 'stopband_edge_freq':np.array([20, 450]), 'passband_edge_max_loss':1, 'stopband_edge_min_loss':10}):
        if method=='bandpass_filtering':
            self.data = bandpass_filter(
                                        data=self.data,
                                        t_data=self.t_data,
                                        passband_edge_freq=params['passband_edge_freq'],
                                        stopband_edge_freq=params['stopband_edge_freq'],
                                        passband_edge_max_loss=params['passband_edge_max_loss'],
                                        stopband_edge_min_loss=params['stopband_edge_min_loss'],
                                        )
        else:
            print('There is no such method.')
    
    def filter_video(self, params_list=[{'kernel_length':10, 'kernel_size':3}]):
        video_data_list_filtered = []
        for i in range(len(self.video_data_list)):
            video_data_list_filtered.append(video_filtering(self.video_data_list[i], kernel_length=params_list[i]['kernel_length'], kernel_size=params_list[i]['kernel_size']))
        self.video_data_list = video_data_list_filtered

    def linescanning_video(self, params_list=[{'position':[70, 80], 'width':70}]):
        video_data_list_scanned = []
        for i in range(len(self.video_data_list)):
            video_data_list_scanned.append(line_scanning(self.video_data_list[i], position=params_list[i]['position'], width=params_list[i]['width']))
        self.video_data_list = video_data_list_scanned

    def savgol_filter_video(self, params_list=[{'window_length':20, 'polyorder':3}]):
        video_data_list_filtered = []
        for i in range(len(self.video_data_list)):
            video_data_list_filtered.append(video_savgol_filtering(self.video_data_list[i], window_length=params_list[i]['window_length'], polyorder=params_list[i]['polyorder']))
        self.video_data_list = video_data_list_filtered

    def standardize_video(self):
        video_data_list_standardized = []
        for i in range(len(self.video_data_list)):
            video_data_list_standardized.append(standardize(self.video_data_list[i]))
        self.video_data_list = video_data_list_standardized
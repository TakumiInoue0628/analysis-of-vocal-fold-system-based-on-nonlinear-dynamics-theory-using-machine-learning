from os.path import dirname, abspath
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
### Move to parent directory
parent_dir = dirname(dirname(abspath(__file__)))
sys.path.append(parent_dir)
from preprocessing.function import *

def preprocess_csv(config):
    cfg = config
    ### Normal preprocessing
    for i in range(len(cfg.DATA_NAME_LIST)):
        ### Load raw csv
        raw_csv_path = cfg.RAW_CSV_DIR + cfg.DATA_NAME_LIST[i] + '.csv'
        data_name_list, data_list = load_csv_data(raw_csv_path, cfg.UPPERCASE_COLUMN)
        ### Save csv
        save_csv_path = cfg.SAVE_CSV_DIR + cfg.DATA_NAME_LIST[i] + '.csv'
        save_csv_data(data_name_list, data_list, save_csv_path)
        ### Flash preprocessing
        if cfg.FLASH_PP:
            ### Get flash point
            stt_idx, _, end_idx, _ = get_flash_csv(save_csv_path, cfg.UPPERCASE_COLUMN)
            ### Remove before flash & after flash
            if cfg.START_FLASH_ONLY: remove_before_after_flash_csv(save_csv_path, save_csv_path, stt_idx, end_idx, True)
            else: remove_before_after_flash_csv(save_csv_path, save_csv_path, stt_idx, end_idx)

def preprocess_video(config):
    cfg = config
    ### Normal preprocessing
    for i in range(len(cfg.DATA_NAME_LIST)):
        ### Load raw video
        raw_video_path = cfg.RAW_VIDEO_DIR + cfg.DATA_NAME_LIST[i] + '.avi'
        ### Save video
        save_video_path = cfg.SAVE_VIDEO_DIR + cfg.DATA_NAME_LIST[i] + '.avi'
        ### Flash preprocessing
        if cfg.FLASH_PP:
            ### Get flash point
            stt_frm, _, end_frm, _ = get_flash_video(raw_video_path, cfg.FLASH_SEARCH_SPAN[0], cfg.FLASH_SEARCH_SPAN[-1], cfg.SAVE_VIDEO_DIR + cfg.DATA_NAME_LIST[i] + '_')
            ### Remove before flash & after flash
            if cfg.STOP_FLASH_ONLY: stt_frm = 0
            remove_before_after_flash_video(raw_video_path, save_video_path, stt_frm, end_frm)

def preprocess_kymogram(config):
    cfg = config
    ### Normal preprocessing
    for i in range(len(cfg.DATA_NAME_LIST)):
        ### Load raw video
        raw_video_path = cfg.RAW_VIDEO_DIR + cfg.DATA_NAME_LIST[i] + '.avi'
        ### Save directry
        save_kymogram_path = cfg.SAVE_KYMOGRAM_DIR + cfg.DATA_NAME_LIST[i] + '.txt'
        save_time_path = cfg.SAVE_KYMOGRAM_DIR + cfg.DATA_NAME_LIST[i] + '_time.txt'
        ### Line-scanning preprocessing
        kymogram, time = line_scanning(raw_video_path, cfg.LINESCANNING_PARAMS_LIST[i]['position'], cfg.LINESCANNING_PARAMS_LIST[i]['width'])
        ### Save kymogram
        np.savetxt(save_kymogram_path, kymogram)
        np.savetxt(save_time_path, time)

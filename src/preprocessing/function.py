import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tqdm


###################### CSV ######################
def load_csv_data(
                csv_path,
                uppercase_column=True,
                ):
    ### load csv
    input_data = np.loadtxt(csv_path, delimiter=',', encoding="utf-8", dtype='unicode')
    data_num = int(input_data.shape[1]/2)
    data_name_list = []
    data_list = []
    ### add time data
    if uppercase_column:
        data_name_list.append('T')
    else:
        data_name_list.append('t')
    data_list.append(input_data[4:, 0].astype(np.float64))
    ### add other data
    for i in range(data_num):
        data_name_list.append(input_data[2, 2*i+1].replace('"', ''))
        data_list.append(input_data[4:, 2*i+1].astype(np.float64))
    return  data_name_list, data_list

def save_csv_data(
                data_name_list,
                data_list,
                save_csv_path,
                csv_index = False,
                    ):
    ### generate dataframe
    df = pd.DataFrame(data=np.array(data_list).T)
    ### add columns
    df.columns = data_name_list
    ### output csv 
    df.to_csv(save_csv_path, index=csv_index)

def get_flash_csv(
                    csv_path,
                    uppercase_column=True,
                    sample_span=None,
                    #save_fig_path,
                    #out_fig=True,
                    ):
    ### load csv
    data_df = pd.read_csv(csv_path)
    if uppercase_column:
        time_data = data_df[['T']].values[:, 0]
        flash_data = data_df[['FLASH']].values[:, 0]   
    else:
        time_data = data_df[['t']].values[:, 0]
        flash_data = data_df[['flash']].values[:, 0]    
    if sample_span is None:
        sample_span = int(time_data.shape[0] / 2)
    #
    '''
    mic_data = data_df[['mic']].values[:, 0]
    plt.figure(figsize=(5, 3))
    plt.plot(time_data, mic_data, lw=0.5, label='mic data')
    plt.plot(time_data, flash_data, lw=0.5, label='flash data')
    plt.xlabel('time')
    plt.legend()
    plt.show()
    '''
    #
    ### get flash point
    exp_start_flash_data = flash_data[:sample_span]
    exp_stop_flash_data = flash_data[flash_data.shape[0]-sample_span:]
    exp_start_flash_diff = np.diff(exp_start_flash_data)
    exp_stop_flash_diff = np.diff(exp_stop_flash_data)
    exp_start_flash_idx = np.argmin(exp_start_flash_diff)
    exp_stop_flash_idx = flash_data.shape[0]-sample_span + np.argmax(exp_stop_flash_diff)
    exp_start_flash_time = time_data[exp_start_flash_idx]
    exp_stop_flash_time = time_data[exp_stop_flash_idx]
    ### plot flash value
    '''
    if out_fig:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.set_title('Experiment start time span')
        plot_x_value = np.arange(0, sample_span)
        ax.plot(plot_x_value, exp_start_flash_data, lw=0.5)
        ax.set_xlabel('sample number')
        ax.set_ylabel('flash')
        ax = fig.add_subplot(122)
        ax.set_title('Experiment stop time span')
        plot_x_value = np.arange(flash_data.shape[0]-sample_span, flash_data.shape[0])
        ax.plot(plot_x_value, exp_stop_flash_data, lw=0.5)
        ax.set_xlabel('sumple number')
        plt.savefig(save_fig_path)    
    '''
    ### results
    print('-------------FLASH POINT-------------')
    print('csv path')
    print(str(csv_path)+'\n')
    print('Experiment start flash point')
    print('sample number |'+str(exp_start_flash_idx))
    print('time          |'+str(exp_start_flash_time)+' s'+'\n')
    print('Experiment stop flash point')
    print('sample number |'+str(exp_stop_flash_idx))
    print('time          |'+str(exp_stop_flash_time)+' s')
    print('-------------------------------------\n')
    return exp_start_flash_idx, exp_start_flash_time, exp_stop_flash_idx, exp_stop_flash_time

def remove_before_after_flash_csv(
                                    csv_path,
                                    save_csv_path,
                                    exp_start_flash_idx,
                                    exp_stop_flash_idx,
                                    exp_start_flash_idx_only=False,
                                    margin=5,
                                    csv_index=False
                                    ):
    ### load csv
    data_df = pd.read_csv(csv_path)
    if exp_start_flash_idx_only: new_data_df = data_df[exp_start_flash_idx+margin:]
    else: new_data_df = data_df[exp_start_flash_idx+margin:exp_stop_flash_idx-margin]
    ### output csv 
    new_data_df.to_csv(save_csv_path, index=csv_index)

###################### VIDEO ######################

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
    if time_span[1]==None: stop_frame = np.argmin(abs(video_t-video_t[-1]))
    else: stop_frame = np.argmin(abs(video_t-time_span[1]))
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
    
def get_flash_video(
                    video_path,
                    frame_span_stt=0,
                    frame_span_end=0,
                    save_fig_dir='',
                    out_fig=True,
                    ):
    ### VideoCapture (get object)
    cap = cv2.VideoCapture(video_path)
    ### get video property
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_span_stt == 0:
        frame_span_stt = int(frame_count / 2)
    if frame_span_end == 0:
        frame_span_end = int(frame_count / 2)
    ### read frames
    print('------------- READ VIDEO FRAMES -------------')
    print('video path                    | '+str(video_path))
    print('Reading frames...')
    exp_start_frames = []
    exp_stop_frames = []
    for i in tqdm.tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret: # read successed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # RGB 3ch --> GRAY 1ch
            ### experiment start flash point
            if i < frame_span_stt:
                exp_start_frames.append(frame)
            ### experiment stop flash point
            if i > frame_count-frame_span_end-1:
                exp_stop_frames.append(frame)
        else : # read failed
            break
    cap.release()
    print('-------------------------------------------\n')
    print('Calculating flash point...\n')  
    ### datatype list --> numpy
    exp_start_frames = np.array(exp_start_frames)
    exp_stop_frames = np.array(exp_stop_frames)
    ### confirm luminance value
    exp_start_mean_luminance = np.mean(exp_start_frames, (1, 2))
    exp_stop_mean_luminance = np.mean(exp_stop_frames, (1, 2))
    exp_start_flash_frame = np.argmax(exp_start_mean_luminance)
    exp_stop_flash_frame = frame_count - frame_span_end + np.argmax(exp_stop_mean_luminance)
    exp_start_flash_time = exp_start_flash_frame / fps
    exp_stop_flash_time = exp_stop_flash_frame / fps 
    ### plot luminance value
    
    if out_fig:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax.set_title('Experiment start time span')
        #plot_x_value = np.arange(0, time_span*fps)
        #ax.plot(plot_x_value, exp_start_mean_luminance, lw=0.5)
        ax.plot(exp_start_mean_luminance, lw=0.5)
        ax.set_xlabel('frame number')
        ax.set_ylabel('luminance value (mean)')
        ax = fig.add_subplot(122)
        ax.set_title('Experiment stop time span')
        #plot_x_value = np.arange(frame_count - time_span*fps, frame_count)
        #ax.plot(plot_x_value, exp_stop_mean_luminance, lw=0.5)
        ax.plot(exp_stop_mean_luminance, lw=0.5)
        ax.set_xlabel('frame number')
        plt.savefig(save_fig_dir+'liminance.png')
    
    ### results
    print('-------------FLASH POINT-------------')
    print('video path')
    print(str(video_path)+'\n')
    print('Experiment start flash point')
    print('frame number |'+str(exp_start_flash_frame))
    print('time         |'+str(exp_start_flash_time)+' s'+'\n')
    print('Experiment stop flash point')
    print('frame number |'+str(exp_stop_flash_frame))
    print('time         |'+str(exp_stop_flash_time)+' s')
    print('-------------------------------------\n')
    return exp_start_flash_frame, exp_start_flash_time, exp_stop_flash_frame, exp_stop_flash_time

def remove_before_after_flash_video(
                                    video_path,
                                    save_video_path,
                                    exp_start_flash_frm,
                                    exp_stop_flash_frm,
                                    margin=5,
                                    ):
    start_frame, stop_frame = exp_start_flash_frm+margin, exp_stop_flash_frm-margin
    ### VideoCapture (get object)
    cap = cv2.VideoCapture(video_path)
    ### get video property
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ### video writter settings
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
    ### read frames
    print('\n------------- PREPROCESS VIDEO -------------')
    print('video path     | '+str(video_path))
    print('frame span     | ['+str(exp_start_flash_frm)+', '+str(exp_stop_flash_frm)+']')
    print('time span (s)  | ['+str(exp_start_flash_frm/fps)+', '+str(exp_stop_flash_frm/fps)+']')
    print('Reading & Converting frames...')
    for i in tqdm.tqdm(range(stop_frame+1)):
        ret, frame = cap.read()
        if ret: # read successed
            if i > int(start_frame-1):
                ### write video
                writer.write(frame)
        else : # read failed
            break
    writer.release()
    cap.release()
    print('-------------------------------------\n')

###################### KYMOGRAM ######################

def line_scanning(video_path, position, width, shooting_time_interval=1/10000, to_GRAY=True,):
    
    ### Load video
    print('Loading video data')
    print('file path | '+video_path)
    ### VideoCapture (get object)
    cap = cv2.VideoCapture(video_path)
    ### get video property
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ### generate video time data
    video_t = np.arange(0, frame_count+1)*shooting_time_interval
    ### decide start load point & end load point
    start_frame = 0
    stop_frame = frame_count
    t_data = video_t[start_frame:stop_frame] 
    ### load video
    frames = []
    for i in tqdm.tqdm(range(stop_frame), desc="Loading", leave=False):
        ret, frame = cap.read()
        if ret: # read successed
            if i > int(start_frame-1):
                ### RGB 3ch --> GRAY 1ch
                if to_GRAY: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ### save frame
                line_scanned_frame = frame[position[1], (position[0]-int(width/2)):(position[0]+int(width/2))]
                frames.append(line_scanned_frame)
        else : # read failed
            break
    cap.release()
    ### convert datatype: list init8 --> numpy float64
    kymogram_data = np.array(frames).astype(np.float64)

    return kymogram_data, t_data
from os.path import dirname, abspath
import sys
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
import matplotlib.animation as animation
### Move to parent directory
parent_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(parent_dir)
### Import module
from src.util import short_time_ft, fft
from src.util import Normalization

class signal():

    def __init__(self, rcParams_dict, savefig=False, save_dir=None, name=None, file_type='png'):
        for key in rcParams_dict.keys():
            plt.rcParams[str(key)] = rcParams_dict[str(key)]
        self.save = savefig
        self.dir = save_dir
        self.name = name
        self.ext = file_type

    def timeseries(self, time, signal, 
                   figsize=(10, 2),
                   title='',
                   xlabel=r'Time [s]', ylabel=r'[]',
                   xformat='%.1f', yformat='%.1f',
                   ylim=(None, None),
                   file_name=''):    
        spec = gridspec.GridSpec(ncols=1, nrows=1, height_ratios=[1], hspace=0)
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(spec[0])
        ax.set_title(title, loc='left', fontsize=plt.rcParams['font.size'])
        ax.plot(time, signal, c='k', lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(xformat))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter(yformat))
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(ylim[0], ylim[-1])

        if self.save: 
            plt.savefig(self.dir+self.name+'_timeseries_'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def timeseries_spectrogram(self, time, signal, stfft_params,
                                figsize=(10, 4),
                                title='',
                                xlabel=r'Time [s]', ylabel=r'[]',
                                xformat='%.1f', yformat='%.1f',
                                ylim=(None, None),
                                file_name=''):    
        spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1, 0.04], height_ratios=[1, 1], wspace=0, hspace=0.55)
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(spec[0])
        ax.set_title(title+' (timeseries)', loc='left', fontsize=plt.rcParams['font.size'])
        ax.plot(time, signal, c='k', lw=1)
        ax.set_ylabel(ylabel)
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(xformat))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter(yformat))
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(ylim[0], ylim[-1])

        ax = fig.add_subplot(spec[2])
        ax.set_title(title+' (spectrogram)', loc='left', fontsize=plt.rcParams['font.size'])
        dt = time[1] - time[0]
        freq, t_stft, intens = short_time_ft(signal, dt, stfft_params['nperseg'])
        p = np.linspace(time[0], time[-1], t_stft.shape[0])
        power = 10*np.log(np.abs(intens))
        img = ax.pcolormesh(p, freq, power, cmap=stfft_params['cmap'], vmin=stfft_params['cmap_lim'][0], vmax=stfft_params['cmap_lim'][1])
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(stfft_params['freq_lim'])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'Frequency [Hz]')
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(xformat))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        
        ax = fig.add_subplot(spec[3])
        ax.axis("off")
        fig.colorbar(img, location='right', ax=ax, shrink=10, label=r'[$10\times\log|x|$]')
        
        if self.save: 
            plt.savefig(self.dir+self.name+'_timeseries_spectrogram_'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def flow_timeseries_spectrogram(self, time, flow, signal, stfft_params,
                                figsize=(10, 6),
                                title='',
                                xlabel=r'Time [s]', ylabel=r'[]',
                                xformat='%.1f', yformat='%.1f',
                                ylim_flow=(None, None), ylim_signal=(None, None),
                                steady_range=[None, None],
                                file_name=''):    
        spec = gridspec.GridSpec(ncols=2, nrows=3, width_ratios=[1, 0.04], height_ratios=[1, 1, 1], wspace=0, hspace=0.55)
        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(spec[0])
        ax.set_title('Flow signal'+' (timeseries)', loc='left', fontsize=plt.rcParams['font.size'])
        ax.plot(time, flow, c='k', lw=1)
        ax.set_ylabel(r'[L/min]')
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(xformat))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter(yformat))
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(ylim_flow[0], ylim_flow[-1])
        if steady_range!=[None, None]: ax.axvline(x=time[steady_range[0]], linestyle='--', c='b', lw=1)
        if steady_range!=[None, None]: ax.axvline(x=time[steady_range[-1]], linestyle='--', c='b', lw=1)

        ax = fig.add_subplot(spec[2])
        ax.set_title(title+' (timeseries)', loc='left', fontsize=plt.rcParams['font.size'])
        ax.plot(time, signal, c='k', lw=1)
        ax.set_ylabel(ylabel)
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(xformat))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter(yformat))
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(ylim_signal[0], ylim_signal[-1])
        if steady_range!=[None, None]: ax.axvline(x=time[steady_range[0]], linestyle='--', c='b', lw=1)
        if steady_range!=[None, None]: ax.axvline(x=time[steady_range[-1]], linestyle='--', c='b', lw=1)

        ax = fig.add_subplot(spec[4])
        ax.set_title(title+' (spectrogram)', loc='left', fontsize=plt.rcParams['font.size'])
        dt = time[1] - time[0]
        freq, t_stft, intens = short_time_ft(signal, dt, stfft_params['nperseg'])
        p = np.linspace(time[0], time[-1], t_stft.shape[0])
        power = 10*np.log(np.abs(intens))
        img = ax.pcolormesh(p, freq, power, cmap=stfft_params['cmap'], vmin=stfft_params['cmap_lim'][0], vmax=stfft_params['cmap_lim'][1])
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(stfft_params['freq_lim'])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'Frequency [Hz]')
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter(xformat))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        if steady_range!=[None, None]: ax.axvline(x=time[steady_range[0]], linestyle='--', c='b', lw=1)
        if steady_range!=[None, None]: ax.axvline(x=time[steady_range[-1]], linestyle='--', c='b', lw=1)
        
        ax = fig.add_subplot(spec[5])
        ax.axis("off")
        fig.colorbar(img, location='right', ax=ax, shrink=10, label=r'[$10\times\log|x|$]')
        
        if self.save: 
            plt.savefig(self.dir+self.name+'_timeseries_spectrogram_'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def timeseries_information_delaycoordinate(self, time, data, I, tau, 
                                               n_plt=3000, style='.',
                                                figsize=(10, 2),
                                                file_name=''):    
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[3.3, 1], wspace=0.5)
        fig = plt.figure(figsize=(10, 2))

        ax1 = fig.add_subplot(spec[0])
        ax1.set_title(r'Timeseries $x(t)$, Mutual information $I(t)$ ', fontsize=plt.rcParams['font.size'])
        ax1.plot(time[:I.shape[0]], data[:I.shape[0]], lw=1, c='k')
        ax1.set_ylabel(r'$x(t)$')
        ax1.set_xlabel(r'Time [s]')
        #plt.legend(loc='upper right', frameon=False, fancybox=False, edgecolor='k')
        ax1.set_xlim(time[0], time[I.shape[0]])
        ax1.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.2f'))
        ax1.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax1.set_ylim(None, np.max(data[:I.shape[0]])+(np.max(data[:I.shape[0]])-np.min(data[:I.shape[0]]))/2)

        ax2 = ax1.twinx()
        ax2.plot(time[:I.shape[0]], I, '--', lw=1, c='b')
        ax2.set_ylabel(r'$I(t)$', c='b')
        ax2.tick_params(axis='y', colors='b')
        ax2.axvline(x=time[tau], linestyle=':', c='r', lw=1.5, label=r'First local minimum $\tau$')
        ax2.set_xlim(time[0], time[I.shape[0]])
        ax2.set_ylim(None, np.max(I[1:])+(np.max(I[1:])-np.min(I[1:]))/2)
        ax2.spines['right'].set_color('b')
        plt.legend(loc='upper right', frameon=False)

        ax = fig.add_subplot(spec[1])
        ax.set_title(r'Delay coordinate', fontsize=plt.rcParams['font.size'])
        ax.plot(data[tau:][:n_plt], data[:-tau][:n_plt], style, markersize=1.0, lw=1.0, c='k')
        ax.set_xlabel(r'$x(t)$')
        ax.set_ylabel(r'$x(t-\tau)$')
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        if self.save: 
            plt.savefig(self.dir+self.name+'_timeseries_information_delaycoordinate_'+file_name+'.'+self.ext, bbox_inches="tight")  
        plt.show()

    def timeseries_attractor_powerspectra(self, time, data, tau, n_plt, 
                                          figsize=(10, 2), file_name='', 
                                          x_lim=[None, None], f_lim=[None, None], p_lim=[0, 500], 
                                          style='.'):
        spec = gridspec.GridSpec(ncols=3, nrows=1, width_ratios=[2.5, 1.5, 2], wspace=0.4)
        fig = plt.figure(figsize=figsize)
        
        ax = fig.add_subplot(spec[0])
        ax.set_title('Timeseries', fontsize=plt.rcParams['font.size'])
        ax.plot(time[:n_plt], data[:n_plt], lw=1.0, c='k')
        ax.set_xlabel(r'Time [s]')
        ax.set_ylabel(r'$x(t)$')
        ax.set_ylim(x_lim[0], x_lim[-1])
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        ax = fig.add_subplot(spec[1])
        ax.set_title(r'Delay coordinate', fontsize=plt.rcParams['font.size'])
        ax.plot(data[tau:][:n_plt], data[:-tau][:n_plt], style, markersize=1.0, lw=1.0, c='k')
        ax.set_xlabel(r'$x(t)$')
        ax.set_ylabel(r'$x(t-\tau)$')
        ax.set_xlim(x_lim[0], x_lim[-1])
        ax.set_ylim(x_lim[0], x_lim[-1])
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        ax.get_yaxis().set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        ax = fig.add_subplot(spec[2])
        ax.set_title('Power spectra', fontsize=plt.rcParams['font.size'])
        freq_org, amp_org = fft(data, time)
        ax.plot(freq_org, amp_org, lw=1.0, c='k')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel(r'Power')
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%d'))
        ax.set_xlim(f_lim[0], f_lim[1])
        ax.set_ylim(p_lim[0], p_lim[1])
        
        if self.save: plt.savefig(self.dir+self.name+'_timeseries_attractor_powerspectra'+'_'+file_name+'.'+self.ext, bbox_inches="tight") 
        plt.show()
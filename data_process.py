import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import hamming, stft, get_window, windows,butter, filtfilt, detrend
import os

class FMCWProcess:
    def __init__(self):
        self.Br = 100e6        # 调制带宽
        # PRT = 1.024e-3    # 调制周期
        self.PRT = 0.3e-3      # 调制周期 (default)
        self.Fs = 500e3        # 采样频率
        self.minR = 2          # 起始距离
        self.maxR = 15         # 终止距离
        self.C = 3e8           # 光速
        self.PRF = 1 / self.PRT     # 重复频率
        self.target_range = 9  # 目标距离 (米)

    def moving_average(self, dc_removed, window_size=5):
        n_range, n_pulse = dc_removed.shape
        trend_removed = np.zeros_like(dc_removed)
        
        for i in range(n_range):
            signal = dc_removed[i, :]
            trend = np.zeros_like(signal)
            
            for j in range(n_pulse):
                left = max(0, j - window_size)
                right = min(n_pulse, j + window_size + 1)
                neighbors = np.concatenate([signal[left:j], signal[j+1:right]])
                trend[j] = np.mean(neighbors) if len(neighbors) > 0 else 0
            
            trend_removed[i, :] = signal - trend
        
        return trend_removed
    
    def read_data(self, file_path):
        mat_data = loadmat(file_path)
        basename = os.path.basename(file_path)
        self.PRT = float(basename.split('-')[2]) * 1e-3
        self.PRF = 1 / self.PRT
        self.Br = float(basename.split('-')[3]) * 1e6
        self.target_range = float(basename.split('-')[4])
        
        self.echo = mat_data['echoes']['channelA'][0][0]   # get I/Q echo signal
        self.echo = self.echo / np.std(self.echo)

        return self.echo

    def pre_processing(self, echo=None, use_filter=True):
        if echo is None:
            echo = self.echo

        self.Nr, self.Na = echo.shape
        echo_positive = echo[:self.Nr//2, :]   # de-negative
        echo_RDC = echo_positive - np.mean(echo_positive, axis=0, keepdims=True)  # de-DC
        if use_filter:
            echo_smooth = self.moving_average(echo_RDC, window_size=5)  # debounce
        else:
            echo_smooth = echo_RDC

        return echo_smooth
    
    def range_FFT(self, echo):
        # time axis
        time = self.Na * self.PRT
        t_axis = np.linspace(0, time, self.Na)

        Nfft = 2 ** (int(np.ceil(np.log2(self.Nr))) + 2)  # 4x zero padding
        # range-dim FFT
        window = hamming(self.Nr//2)[:, np.newaxis]
        echo_FFT = np.fft.fftshift(np.fft.fft(echo * window, n=Nfft, axis=0), axes=0)
        echo_FFT = echo_FFT[Nfft//2:, :]  # positive frequency

        # new distance index
        minP = round(self.Br / (self.PRT/2) * (2 * self.minR / self.C) / (self.Fs / Nfft))
        maxP = round(self.Br / (self.PRT/2) * (2 * self.maxR / self.C) / (self.Fs / Nfft))

        Nrp = maxP - minP + 1
        # Rr = np.linspace(self.minR, self.maxR, Nrp)

        echo_FFT = echo_FFT[minP-1:maxP, :]  # crop distance window

        return echo_FFT
    
    def get_RD(self, echo):
        Nfft = 2 ** (int(np.ceil(np.log2(self.Na))) + 2)  # 4x zero padding
        window_d = hamming(self.Na)[np.newaxis, :]
        echo_RD = np.fft.fftshift(np.fft.fft(echo * window_d, n=Nfft,axis=1), axes=1)

        # Echo_RD_mag = 20 * np.log10(np.abs(Echo_RD[:Nr//2, :]) + 1e-6)
        # RD_amp = np.abs(Echo_RD_mag)
        # RD_dB = 20 * np.log10(RD_amp + 1e-10)  
        # RD_dB = 20 * np.log10(np.abs(Echo_RD) + 1e-6)  
        echo_RD = np.abs(echo_RD)

        return echo_RD
    
    def get_STFT(self, echo):
        Nfft = 2 ** (int(np.ceil(np.log2(self.Nr))) + 2)
        minP = round(self.Br / (self.PRT/2) * (2 * self.minR / self.C) / (self.Fs / Nfft))
        maxP = round(self.Br / (self.PRT/2) * (2 * self.maxR / self.C) / (self.Fs / Nfft))
        Nrp = maxP - minP + 1
        stft_target = round((self.target_range - self.minR) / (self.maxR - self.minR) * (Nrp - 1))

        target_signal = echo[stft_target, :]

        nperseg = 64
        noverlap = 48
        nfft = 256
        win = windows.gaussian(nperseg, std=nperseg//6)

        f, t, Zxx = stft(target_signal, fs=self.PRF, window=win, nperseg=nperseg, 
                        noverlap=noverlap, nfft=nfft, return_onesided=False)
        # f, t, Zxx = stft(target_signal, fs=PRF, nperseg=nperseg, 
        #                  noverlap=noverlap, nfft=nfft, return_onesided=False)

        Zxx = np.fft.fftshift(Zxx, axes=0)
        f = np.fft.fftshift(f)

        # power = 20 * np.log10(np.abs(Zxx) + 1e-10)
        power = np.abs(Zxx)

        # plt.pcolormesh(t, f, power, shading='gouraud', cmap='jet')
        return f, t, Zxx
    
    def get_RD(self, echo):
        Nfft = 2 ** (int(np.ceil(np.log2(self.Na))) + 2)  # 4x零填充
        window_d = hamming(self.Na)[np.newaxis, :]
        echo_RD = np.fft.fftshift(np.fft.fft(echo * window_d, n=Nfft,axis=1), axes=1)

        # Echo_RD_mag = 20 * np.log10(np.abs(Echo_RD[:Nr//2, :]) + 1e-6)
        # RD_amp = np.abs(Echo_RD_mag)
        # RD_dB = 20 * np.log10(RD_amp + 1e-10)  
        # RD_dB = 20 * np.log10(np.abs(Echo_RD) + 1e-6)  # 幅值转换为dB
        return echo_RD

    def signal_vis(self, signal):
        # return 20 * np.log10(np.abs(signal) + 1e-10)
        return np.abs(signal)

    def process(self, file_path):
        echo_raw = self.read_data(file_path)
        echo_smooth = self.pre_processing(echo_raw)
        echo_FFT = self.range_FFT(echo_smooth)
        STFT_f, STFT_t, STFT_Zxx = self.get_STFT(echo_FFT)
        echo_RD = self.get_RD(echo_smooth)
        return self.signal_vis(echo_FFT), self.signal_vis(echo_RD), STFT_f, STFT_t, self.signal_vis(STFT_Zxx)

PreProcess = FMCWProcess()
echo_FFT, echo_RD, STFT_f, STFT_t, STFT_Zxx = PreProcess.process('06-2023.5.8-0.3-100-9-K (1).mat')

# vis distance-frequency
plt.figure()
plt.imshow(echo_FFT, aspect='auto')
plt.xlabel(r"freq")
plt.ylabel(r"distance(m)")
plt.title(r"distance-range")
plt.gca().invert_yaxis()
plt.colorbar()


# vis STFT time-freq map
plt.figure()
plt.pcolormesh(STFT_t, STFT_f, STFT_Zxx, shading='gouraud', cmap='jet')
plt.title(f"STFT")
plt.ylabel("F (kHz)")
# plt.ylim([-PRF/2, PRF/2])
plt.xlabel("T (s)")
plt.colorbar(label='S')
plt.tight_layout()


# vis RD range-Doppler map
plt.figure()
plt.imshow(echo_RD, aspect='auto', cmap='jet')
plt.xlabel(r"T")
plt.ylabel(r"D(m)")
plt.title(r"RD")
plt.gca().invert_yaxis()
plt.colorbar()

plt.show()

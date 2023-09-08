from tftb.generators import amgauss, fmlin, fmconst
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from tftb.processing import WignerVilleDistribution, inst_freq, plotifl
from scipy.signal import hamming, cwt, ricker
from tftb.processing import Spectrogram

N = 3000
fs = 1000
dt = 1 / fs
t = np.arange(0, N*dt, dt)

# Run a FFT analysis to get an idea of the frequency components. 
# Reflect on the results of this analysis
def fft(signal):
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / fs)

    plt.figure(figsize=(8, 4))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title("FFT of the Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim(-50, 50)
    plt.show()



## Which signal processing technique is best for your signal (FFT, STFT, WVT, WT, HT)?

# STFT
def stft(signal):
    fwindow = hamming(1000)
    spec = Spectrogram(signal, n_fbins=128, fwindow=fwindow)
    spec.run()
    spec.plot(kind="contour", threshold=0.1, show_tf=False)

# WVT
def wvt(signal): 
    n_points = 128
    fmin, fmax = 0.0, 0.5
    wvd = WignerVilleDistribution(signal)
    wvd.run()
    wvd.plot(kind='contour', extent=[0, n_points, fmin, fmax])


# Hilbert Transform / Instantaneous Frequency
def ht(signal):
    ifr = inst_freq(signal)[0]
    plotifl(np.linspace(0,len(ifr), len(ifr)), ifr)

# Wavelet Transform
def wt(signal):
    widths = np.arange(1, 31)
    cwtmatr = cwt(signal, ricker, widths)
    cwtmatr_yflip = np.flipud(cwtmatr)
    plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()
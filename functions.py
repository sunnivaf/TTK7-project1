from tftb.generators import amgauss, fmlin, fmconst
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from tftb.processing import WignerVilleDistribution, inst_freq, plotifl
from scipy.signal import hamming
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

## How to decide the window size if STFT or WT is going to be used?
# Since we are dealing with a stationary signal a longer window is preferred to capture
# more frequency details. A longer window is also preferred as the frequency components are of low frequency.


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

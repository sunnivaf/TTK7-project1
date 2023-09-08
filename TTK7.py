from tftb.generators import amgauss, fmlin, fmconst
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from tftb.processing import WignerVilleDistribution, inst_freq, plotifl
from scipy.signal import hamming
from tftb.processing import Spectrogram



# Define the time values
# t = np.linspace(0, 3000, 1000)  # Time from 0 to 1 second, with 1000 points
N = 3000
fs = 1000
dt = 1 / fs
t = np.arange(0, N*dt, dt)

# Print the result
print("Sampling Frequency:", fs, "Hz")

# Define the frequencies of the three components
frequencies = [5, 12, 15]  # Frequencies in Hz

# Create the sinusoidal waveforms for all three components as a vector
components = np.sin(2 * np.pi * np.outer(frequencies, t))

# Combine the three components to create the signal
signal = np.sum(components, axis=0)

# Plot the signal
# plt.figure(figsize=(10, 6))
# plt.plot(t, signal)
# plt.title('Signal with 5Hz, 12Hz, and 15Hz Components')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()


## Analyse the final signal: stationary or non-stationary
# The final signal is stationary because it doesn't change over time

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



# # Add an offset and repeat the analysis
# offset = 2
# signal_o = signal + offset

# plt.figure(figsize=(10,4))
# plt.plot(t, signal_o)
# plt.title('Signal with offset')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# # Add white noise and repeat the analysis
# mean = 0
# std = 0.5
# num_samples = len(t)
# white_noise = np.random.normal(mean, std, size=num_samples)

# signal_wn = signal + white_noise
# plt.figure(figsize=(10,4))
# plt.plot(t, signal_wn)
# plt.title('Signal with white noise')
# plt.show()

# # Add a linearly time varying frequency component and repeat the analysis (frequency=kt)
# fmin, fmax = 0.0, 0.05

# lin_var_freq_comp, _ = fmlin(len(t), fmin, fmax)
# lin_var_freq_comp = np.real(lin_var_freq_comp)
# signal_lfq = signal + lin_var_freq_comp

# plt.figure(figsize=(10,4))
# plt.plot(t, signal_lfq)
# plt.title('Signal with Linearly varying frequency component')
# plt.show()

# # Add an offset and white noise and repeat the analysis
# signal_wn_and_o = signal + white_noise + offset
# plt.figure(figsize=(10,4))
# plt.plot(t, signal_wn)
# plt.title('Signal with offset and white noise')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()
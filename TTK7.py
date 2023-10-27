from tftb.generators import fmlin
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from functions import *

## Analyse the final signal: stationary or non-stationary
# The signal is stationary since it is not changing over time

## How to decide the window size if STFT or WT is going to be used?
# Since we are dealing with a stationary signal a longer window is preferred to capture
# more frequency details. A longer window is also preferred as the frequency components are of low frequency.

# Sample rate and duration of the signal
sample_rate = 1000  # Hz
duration = 3.0  # seconds
num_samples = int(sample_rate * duration)

# Create a time vector
t = np.linspace(0, duration, num_samples, endpoint=False)

# Create your signal with three components (5 Hz, 12 Hz, 15 Hz)
component_1 = np.sin(2 * np.pi * 5 * time)
component_2 = np.sin(2 * np.pi * 12 * time)
component_3 = np.sin(2 * np.pi * 15 * time)
signal = component_1 + component_2 + component_3

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Signal with 5Hz, 12Hz, and 15Hz Components')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# freq, fft_result = fft(signal)
# f, t, Sxx = stft(signal)
# wvd, n_points, fmin, fmax = wvt(signal)
# cwtmatr = wt(signal)
# analytic_signal, amplitude_envelope, instantaneous_frequency = ht(signal)

# Add an offset and repeat the analysis
offset = 2
signal_o = signal + offset

# plt.figure(figsize=(10,4))
# plt.plot(t, signal_o)
# plt.title('Signal with offset')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# fft(signal_o)
# stft(signal_o)
# wvt(signal_o)
# wt(signal_o)
# ht(signal_o)

# Add white noise and repeat the analysis
mean = 0
std = 3
white_noise = np.random.normal(mean, std, size=(num_samples,))

signal_wn = signal + white_noise
# plt.figure(figsize=(10,4))
# plt.plot(t, signal_wn)
# plt.title('Signal with white noise')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# fft(signal_wn)
# stft(signal_wn)
# wvt(signal_wn)
# wt(signal_wn)
# ht(signal_wn)

# Add a linearly time varying frequency component and repeat the analysis (frequency=kt)
fmin, fmax = 0.0, 0.05

lin_var_freq_comp, _ = fmlin(num_samples, fmin, fmax)
lin_var_freq_comp = np.real(lin_var_freq_comp)
signal_lfq = signal + lin_var_freq_comp

# plt.figure(figsize=(10,4))
# plt.plot(t, signal_lfq)
# plt.title('Signal with Linearly varying frequency component')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# fft(signal_lfq)
# stft(signal_lfq)
# wvt(signal_lfq)
# wt(signal_lfq)
# ht(signal_lfq)

# Add an offset and white noise and repeat the analysis
signal_wn_and_o = signal + white_noise + offset
# plt.figure(figsize=(10, 4))
# plt.plot(t, signal_wn_and_o)
# plt.title('Signal with offset and white noise')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# fft(signal_wn_and_o)
# stft(signal_wn_and_o)
# wvt(signal_wn_and_o)
# wt(signal_wn_and_o)
# ht(signal_wn_and_o)

labels = ['Original', 'Offset', 'White Noise', 'Linearly Time Varying Frequency', 'White Noise and Offset']
signals = [signal, signal_o, signal_wn, signal_lfq, signal_wn_and_o]
plot_originals(signals, labels)

ffts = [fft(signal), fft(signal_o), fft(signal_wn), fft(signal_lfq), fft(signal_wn_and_o)]
plot_ffts(ffts, labels)

stfts = [stft(signal), stft(signal_o), stft(signal_wn), stft(signal_lfq), stft(signal_wn_and_o)]
plot_stfts(stfts, labels)

hts = [ht(signal), ht(signal_o), ht(signal_wn), ht(signal_lfq), ht(signal_wn_and_o)]
plot_hts(signal, hts, labels)

wts = [wt(signal), wt(signal_o), wt(signal_wn), wt(signal_lfq), wt(signal_wn_and_o)]
plot_wts(wts, labels)

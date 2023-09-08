from tftb.generators import amgauss, fmlin, fmconst
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from tftb.processing import WignerVilleDistribution, inst_freq, plotifl
from scipy.signal import hamming
from tftb.processing import Spectrogram
from functions import *

## Analyse the final signal: stationary or non-stationary
# The signal is stationary since it is not changing over time

## How to decide the window size if STFT or WT is going to be used?
# Since we are dealing with a stationary signal a longer window is preferred to capture
# more frequency details. A longer window is also preferred as the frequency components are of low frequency.

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

fft(signal)

# Add an offset and repeat the analysis
offset = 2
signal_o = signal + offset

plt.figure(figsize=(10,4))
plt.plot(t, signal_o)
plt.title('Signal with offset')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

fft(signal_o)
stft(signal)
wvt(signal_o)
wt(signal_o)
ht(signal_o)

# Add white noise and repeat the analysis
mean = 0
std = 0.5
num_samples = len(t)
white_noise = np.random.normal(mean, std, size=num_samples)

signal_wn = signal + white_noise
plt.figure(figsize=(10,4))
plt.plot(t, signal_wn)
plt.title('Signal with white noise')
plt.show()

fft(signal_wn)
stft(signal_wn)
wvt(signal_wn)
wt(signal_wn)
ht(signal_wn)

# Add a linearly time varying frequency component and repeat the analysis (frequency=kt)
fmin, fmax = 0.0, 0.05

lin_var_freq_comp, _ = fmlin(len(t), fmin, fmax)
lin_var_freq_comp = np.real(lin_var_freq_comp)
signal_lfq = signal + lin_var_freq_comp

plt.figure(figsize=(10,4))
plt.plot(t, signal_lfq)
plt.title('Signal with Linearly varying frequency component')
plt.show()

fft(signal_lfq)
stft(signal_lfq)
wvt(signal_lfq)
wt(signal_lfq)
ht(signal_lfq)

# Add an offset and white noise and repeat the analysis
signal_wn_and_o = signal + white_noise + offset
plt.figure(figsize=(10, 4))
plt.plot(t, signal_wn_and_o)
plt.title('Signal with offset and white noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

fft(signal_wn_and_o)
stft(signal_wn_and_o)
wvt(signal_wn_and_o)
wt(signal_wn_and_o)
ht(signal_wn_and_o)
from tftb.generators import amgauss, fmlin, fmconst
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from tftb.processing import WignerVilleDistribution, inst_freq, plotifl
from scipy.signal import hamming, cwt, ricker, spectrogram
from scipy.signal import hilbert

# Sample rate and duration of the signal
sample_rate = 1000  # Hz
duration = 3.0  # seconds
num_samples = int(sample_rate * duration)

# Create a time vector
time = np.linspace(0, duration, num_samples, endpoint=False)

# Run a FFT analysis to get an idea of the frequency components. 
# Reflect on the results of this analysis
def fft(signal):
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(num_samples, 1 / sample_rate)

    # Plot the magnitude spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(fft_result))
    plt.title('FFT Analysis')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(-20, 20)  # Limit the x-axis to show frequencies up to 20 Hz
    plt.show()


## Which signal processing technique is best for your signal (FFT, STFT, WVT, WT, HT)?

# STFT
def stft(signal):
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=1000)
    plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# WVT
def wvt(signal): 
    n_points = 128
    fmin, fmax = 0.0, 0.5
    wvd = WignerVilleDistribution(signal)
    wvd.run()
    wvd.plot(kind='contour', extent=[0, n_points, fmin, fmax])


# Hilbert Transform / Instantaneous Frequency
def ht(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * sample_rate)
    
    plt.figure(figsize=(12, 6))
    plt.title('Hilbert Transform')

    plt.subplot(211)
    plt.plot(time, signal, label='Original Signal')
    plt.plot(time, amplitude_envelope, label='Instantaneous Amplitude')
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(time[1:], instantaneous_frequency , label='Instantaneous Frequency')
    plt.title('Instantaneous Frequency')
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Wavelet Transform
def wt(signal):
    widths = np.arange(1, 31)
    cwtmatr = cwt(signal, ricker, widths)
    cwtmatr_yflip = np.flipud(cwtmatr)
    
    # Plot the CWT matrix
    plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    
    # Add axis labels
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Wavelet Transform')
    
    plt.show()
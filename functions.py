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
plot_default = False

# Run a FFT analysis to get an idea of the frequency components. 
# Reflect on the results of this analysis
def fft(signal):
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(num_samples, 1 / sample_rate)

    if plot_default:
        # Plot the magnitude spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(freq, np.abs(fft_result))
        plt.title('FFT Analysis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(-20, 20)  # Limit the x-axis to show frequencies up to 20 Hz
        plt.show()
    return (freq, np.abs(fft_result))


## Which signal processing technique is best for your signal (FFT, STFT, WVT, WT, HT)?

# STFT
def stft(signal):
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=1000)
    if plot_default:
        plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    return f, t, np.abs(Sxx)

# WVT
def wvt(signal): 
    n_points = 128
    fmin, fmax = 0.0, 0.5
    wvd = WignerVilleDistribution(signal)
    if plot_default:
        wvd.run()
        wvd.plot(kind='contour', extent=[0, n_points, fmin, fmax])
    return wvd, n_points, fmin, fmax

# Hilbert Transform / Instantaneous Frequency
def ht(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * sample_rate)
    
    if plot_default:
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
    return analytic_signal, amplitude_envelope, instantaneous_frequency


# Wavelet Transform
def wt(signal):
    widths = np.arange(1, 31)
    cwtmatr = cwt(signal, ricker, widths)
    cwtmatr_yflip = np.flipud(cwtmatr)
    if (plot_default):
        # Plot the CWT matrix
        plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
                vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        
        # Add axis labels
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.title('Wavelet Transform')
        
        plt.show()
    return cwtmatr

def plot_originals(signals, labels):
    plt.figure(figsize=(8, 2*len(labels)))
    for idx, (signal, label) in enumerate(zip(signals, labels)):
        plt.subplot(len(labels), 1, idx+1)
        plt.plot(time, signal)
        plt.title(label)
        if (idx == len(labels)-1):
            plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        ax = plt.gca()
        ax.set_xticklabels([])
        plt.grid(True)
    
    plt.subplots_adjust(top=0.962, bottom=0.055, left=0.10, right=0.95, hspace=0.20,
                    wspace=0.219)
    plt.show()

def plot_ffts(ffts, labels):
    plt.figure(figsize=(8, 2*len(labels)))
    # plt.suptitle('FFT Analysis')
    for idx, (fft, label) in enumerate(zip(ffts, labels)):
        freq, fft_result = fft
        plt.subplot(len(labels), 1, idx+1)
        plt.plot(freq, fft_result)
        plt.title(label)
        if (idx == len(labels)-1):
            plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        ax = plt.gca()
        ax.set_xticklabels([])
        plt.grid(True)
        plt.xlim(-20, 20)  # Limit the x-axis to show frequencies up to 20 Hz
    
    plt.subplots_adjust(top=0.962, bottom=0.055, left=0.10, right=0.95, hspace=0.20,
                    wspace=0.219)
    plt.show()

def plot_stfts(stfts, labels):
    plt.figure(figsize=(8, 2*len(labels)))
    
    for idx, (stft, label) in enumerate(zip(stfts, labels)):
        f, t, Sxx = stft
        plt.subplot(len(labels), 1, idx+1)
        plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
        plt.title(label)
        plt.ylabel('Frequency [Hz]')
        if (idx == len(labels)-1):
            plt.xlabel('Time [sec]')
        ax = plt.gca()
        ax.set_xticklabels([])
        plt.grid(True)
        plt.ylim(0, 40)
    
    plt.subplots_adjust(top=0.962, bottom=0.055, left=0.10, right=0.95, hspace=0.20,
                    wspace=0.219)
    plt.show()

def plot_hts(signal, ht, labels):
    plt.figure(figsize=(14, 2*len(labels)))
    # plt.suptitle('FFT Analysis')
    for idx, (ht, label) in enumerate(zip(ht, labels)):
        analytic_signal, amplitude_envelope, instantaneous_frequency= ht
        plt.subplot(len(labels), 2, (2*idx+1))
        
        plt.plot(time, signal, label='Original Signal')
        plt.plot(time, amplitude_envelope, label='Instantaneous Amplitude')
        plt.title(label)
        ax = plt.gca()
        ax.set_xticklabels([])
        plt.grid(True)
        if (idx == len(labels)-1):
            plt.xlabel('Time [sec]')

        plt.subplot(len(labels), 2, (2*idx+1)+1)
        plt.plot(time[1:], instantaneous_frequency , label='Instantaneous Frequency')
        plt.title(label)
        ax = plt.gca()
        ax.set_xticklabels([])
        plt.grid(True)
        if (idx == len(labels)-1):
            plt.xlabel('Time [sec]')
        plt.tight_layout()
    
    plt.subplots_adjust(top=0.962, bottom=0.055, left=0.10, right=0.95, hspace=0.20,
                    wspace=0.219)
    plt.show()

def plot_wts(wts, labels):
    plt.figure(figsize=(8, 2*len(labels)))
    
    for idx, (wt, label) in enumerate(zip(wts, labels)):
        cwtmatr = wt
        cwtmatr_yflip = np.flipud(cwtmatr)

        plt.subplot(len(labels), 1, idx+1)
        plt.imshow(cwtmatr_yflip, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
                vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        
        
        plt.ylabel('Scale')
        plt.title(label)
        if (idx == len(labels)-1):
            plt.xlabel('Time')
        ax = plt.gca()
        ax.set_xticklabels([])
        # plt.grid(True)
        # plt.ylim(0, 40)
    
    plt.subplots_adjust(top=0.962, bottom=0.055, left=0.10, right=0.95, hspace=0.20,
                    wspace=0.219)
    plt.show()
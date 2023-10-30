from tftb.generators import amgauss, fmlin, fmconst
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from tftb.processing import WignerVilleDistribution, inst_freq, plotifl
from scipy.signal import hamming, cwt, ricker, spectrogram, hilbert, stft
import pywt
from pyhht.visualization import plot_imfs
from pyhht import EMD
#from JD_utils import inst_freq, extr, get_envelops

# Sample rate and duration of the signal
sample_rate = 1000  # Hz
duration = 3.0  # seconds
num_samples = int(sample_rate * duration)

# Create a time vector
time = np.linspace(0, duration, num_samples, endpoint=False)
plot_default = True

# Run a FFT analysis to get an idea of the frequency components. 
# Reflect on the results of this analysis
def fft(signal, num_samples, sample_rate, plot=True):
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(num_samples, 1 / sample_rate)

    if plot:
        # Plot the magnitude spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(freq, np.abs(fft_result))
        plt.title('FFT Analysis')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.xlim(0, 50)  # Limit the x-axis to show frequencies up to 20 Hz
        plt.show()
    return (freq, np.abs(fft_result))


## Which signal processing technique is best for your signal (FFT, STFT, WVT, WT, HT)?

# STFT
def plot_stft(signal, sample_rate, nperseg=1000, noverlap=250, plot=False):
    f, t, Sxx = stft(signal, fs = sample_rate, nperseg=nperseg, noverlap=noverlap)

    if plot:
        plt.pcolormesh(t, f, np.abs(Sxx), shading='gouraud')
        plt.colorbar(label='Magnitude [dB]')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim([0, 50])
        plt.show()
    return f, t, np.abs(Sxx)

# WVT
def wvt(signal): 
    # Compute the Wigner-Ville distribution without frequency normalization
    wvd = WignerVilleDistribution(signal)
    wvd.run()

    if plot_default:
        wvd.plot()
        wvd.plot(kind='contour')
    return wvd


#alternative
# def wvt(signal): 
#     # Compute the Wigner-Ville distribution without frequency normalization
#     wvd = WignerVilleDistribution(signal, compute_post_transform=False)
#     tfr_wvd, _, _ = wvd.run()

#     if plot_default:
#         # Plot the Wigner-Ville distribution for positive frequencies
#         plt.figure(figsize=(10, 6))
#         plt.imshow(np.abs(tfr_wvd), origin='lower', aspect='auto', cmap='viridis')
#         plt.xlabel('Time')
#         plt.ylabel('Frequency')
#         plt.title('Wigner-Ville Distribution')
#         plt.colorbar(label='Magnitude')
#         plt.ylim([0, 200])
#         plt.show()
#     return wvd

# Hilbert Transform / Instantaneous Frequency
def ht(t, signal, sample_rate, plot=True):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * sample_rate)
    
    if plot:
        plt.figure()

        plt.subplot(211)
        plt.plot(t, signal, label='Original Signal')
        plt.plot(t, amplitude_envelope, label='Instantaneous Amplitude')
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.xlim()
        plt.grid(True)

        plt.subplot(212)
        plt.plot(t[1:], instantaneous_frequency , label='Instantaneous Frequency')
        plt.title('Instantaneous Frequency')
        plt.xlabel('Time (s)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    return amplitude_envelope, instantaneous_frequency

# Hilbert Huang Transform
def hht(t, signal, plot=True):
    modes = signal;
    decomposer = EMD(modes);
    imfs = decomposer.decompose();
    if plot:
        plot_imfs(modes, imfs, t) ;

    # for imf in imfs[0:len(imfs)-1]:
    #     amplitude_envelope, instantaneous_frequency = ht(imf)

    return imfs

# Wavelet Transform
def wt(t, signal, sample_rate, w=6.0):
    freq = np.linspace(1, sample_rate/2, 100)
    widths = w*sample_rate / (2*freq*np.pi)
    
    cwtm, freqs = pywt.cwt(signal, widths, 'morl')
    
    if (plot_default):
        freq = np.linspace(1, sample_rate/2, 100)
        plt.pcolormesh(t, freq, np.abs(cwtm), cmap='viridis', shading='gouraud')
        plt.colorbar(label='Magnitude [dB]')
        # Add axis labels
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Wavelet Transform')
        plt.ylim([0, 200])

        plt.show()
    return cwtm


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
        plt.xlim(0, 20)  # Limit the x-axis to show frequencies up to 20 Hz
    
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
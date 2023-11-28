import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft
import mne
mne.set_log_level('error')
# from pyhht.visualization import plot_imfs
from pyhht import EMD
try:
    from PyEMD import EEMD, CEEMDAN
except:
    print("Could not load PyEMD")


# Define your frequency bands
frequency_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45)
}


def plot_imfs(imfs, t, signal):
    plt.figure(figsize=(12, 3*(len(imfs)+1)))
    plt.subplot(len(imfs)+1, 1, 1)
    plt.plot(t, signal)
    plt.ylabel('Signal')
    plt.grid(True)
    for idx, imf in enumerate(imfs):
        plt.subplot(len(imfs)+1, 1, idx+2)
        plt.plot(t, imf)
        plt.ylabel(f'imf{idx+1}')
        plt.grid(True)
    plt.show()


def ceemdan(t, signal, trials=100, noise_width=0.05, plot=False):
    modes = signal
    ceemdan = CEEMDAN(trials=trials, noise_width=noise_width)
    eIMFs = ceemdan(modes, t)

    if plot:
        plot_imfs(eIMFs, t, modes)

    return eIMFs

def ERP(raw, event_dict):
    events = mne.find_events(raw, verbose=False)

    sampling_rate = raw.info['sfreq']

    # Epoching settings
    tmin =  -.200  # start of each epoch (in sec)
    tmax =  0.600  # end of each epoch (in sec)
    baseline = (tmin, 0)
    reject = dict(
              eeg=20e-6      # unit: V (EEG channels)
              )
    
    # Create epochs
    epochs = mne.Epochs(raw,
                        events, event_dict,
                        tmin, tmax,
                        baseline=baseline, 
                        #reject=reject,
                        preload=True
                    ) 
    
    fig = mne.viz.plot_events(
    events, event_id=event_dict, sfreq=raw.info["sfreq"], first_samp=raw.first_samp)

    events = epochs.events[:, 0]
    # Convert event sample numbers to time in seconds
    event_times_in_seconds = events / sampling_rate

    # Now, event_times_in_seconds contains the start time of each epoch in seconds
    for idx, event_time in enumerate(event_times_in_seconds):
        print(f"Epoch {idx} starts at {event_time} seconds in the raw data.")

    return epochs, event_times_in_seconds


def freq_band_power(epochs):
    psds = {band: [] for band in frequency_bands} 
    for band, (fmin, fmax) in frequency_bands.items():
        # Compute PSD for the current band
        psd = epochs.compute_psd(fmin=fmin, fmax=fmax)
        psd_mean = psd.get_data().mean()
        # Append the PSD to the list
        psds[band].append(psd_mean)

    return psds


def plot_freq_band_powers(psds_epoch1, psds_epoch2, frequency_bands, channel):
    """
    Plot the frequency band powers for two epochs side by side.

    :param psds_epoch1: Dictionary of mean power spectral densities for epoch 1
    :param psds_epoch2: Dictionary of mean power spectral densities for epoch 2
    :param frequency_bands: Dictionary of frequency bands
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the x-axis labels (frequency bands)
    labels = list(frequency_bands.keys())

    # Get the mean PSD values for each band
    values_epoch1 = [psds_epoch1[band][0] for band in labels]
    values_epoch2 = [psds_epoch2[band][0] for band in labels]

    # Set the positions of the bars
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], values_epoch1, width=0.4, label='Placebo')
    ax.bar([i + 0.2 for i in x], values_epoch2, width=0.4, label='Alcohol')

    # Adding labels and title
    ax.set_xlabel('Frequency Bands')
    ax.set_ylabel('Mean Power Spectral Density')
    ax.set_title(f'{channel} - Frequency Band Powers Across Epochs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Show the plot
    plt.show()

def rfft(signal, num_samples, sample_rate):
    fft_result = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(num_samples, 1 / sample_rate)

    return (freq, np.abs(fft_result))

def fft(signal, num_samples, sample_rate):
    fft_result = np.fft.fft(signal)
    freq = np.fft.fftfreq(num_samples, 1 / sample_rate)

    return (freq, np.abs(fft_result))

def plot_imfs(imfs, t, signal):
    plt.figure(figsize=(8, 2*(len(imfs)+1)))
    plt.subplot(len(imfs)+1, 1, 1)
    plt.plot(t, signal)
    plt.ylabel('Signal')
    plt.grid(True)
    for idx, imf in enumerate(imfs):
        plt.subplot(len(imfs)+1, 1, idx+2)
        plt.plot(t, imf)
        plt.ylabel(f'imf{idx+1}')
        plt.grid(True)
    plt.show()

def plot_imfs_together(placebo, alcohol, t, s1, s2, indices, event_start_line=True, channel=""):
    plt.figure(figsize=(9, 2*(len(indices)+1)))
    plt.subplot(len(indices)+1, 1, 1)
    plt.plot(t, s1, label='Placebo')
    plt.plot(t, s2, label='Alcohol')
    plt.ylabel('Signal')
    plt.legend()
    plt.grid(True)
    for i, idx in enumerate(indices):
        plt.subplot(len(indices)+1, 1, i+2)
        if event_start_line:
            plt.axvline(x = 0, color = 'r', linestyle='dashed',alpha=0.55)
        plt.plot(t, placebo[idx])
        plt.plot(t, alcohol[idx])
        plt.ylabel(f'imf{idx}')
        plt.grid(True)
    plt.subplots_adjust(bottom=0.1, top=0.94)
    plt.suptitle(f"{channel} Channel")
    plt.show()   

def compute_stft(signal, sample_rate, nperseg=1000, noverlap=250, plot=False):
    f, t, Sxx = stft(signal, fs=sample_rate,
                     nperseg=nperseg, noverlap=noverlap)
    return f, t, np.abs(Sxx)

def plot_stft_imf(imfs, sample_rate, nperseg=200, noverlap=0, ylim_min=0, ylim_max=50):
    filt_imfs = imfs[0:len(imfs) - 1]

    plt.figure(figsize=(12, 4 * len(filt_imfs)))

    # Create subplots
    for idx, imf in enumerate(filt_imfs):
        f, dt, Sxx = compute_stft(imf, sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap, plot=False)
        ax = plt.subplot(len(filt_imfs), 1, idx + 1)
        pcm = ax.pcolormesh(dt, f, np.abs(Sxx), shading='gouraud')  # Remove vmin and vmax here
        plt.ylabel('f [Hz]')
        plt.ylim([ylim_min, ylim_max])

        # Calculate the colorbar position
        cbar_x = 1.02  # Adjust this value as needed to position the colorbar correctly
        cbar_width = 0.02
        cbar_height = 0.7 / len(filt_imfs)
        cbar_y = 0.11 + 0.8 * (idx / len(filt_imfs))

        cbar_ax = plt.gcf().add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        plt.colorbar(pcm, cax=cbar_ax, label='Magnitude [dB]')

    # Add a common xlabel
    plt.xlabel('Time [sec]')
    plt.subplots_adjust(right=0.98)  # Adjust the position of individual colorbars
    plt.show()


ylim_min = 0
ylim_max = 50

def plot_stft_imfs(sample_rate, placebo, alcohol, t, s1, s2, p_indices, a_indices, channel="", nperseg=200, noverlap=0, event_start_line=True):

    plt.figure(figsize=(10, 3*len(p_indices)))
    for i in range(len(p_indices)):
        f0, dt, Sxx0 = compute_stft(placebo[p_indices[i]], sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap, plot=False)
        f1, dt, Sxx1 = compute_stft(alcohol[a_indices[i]], sample_rate=sample_rate, nperseg=nperseg, noverlap=noverlap, plot=False)
        vmax = np.amax([np.amax(Sxx0), np.amax(Sxx1)])
        dt = dt-0.2

        ax = plt.subplot(len(p_indices)+1, 2, (2*i)+1)
        if event_start_line:
            plt.axvline(x = 0, color = 'w', linestyle='dashed',alpha=0.35)
        
        pcm = ax.pcolormesh(dt, f0, np.abs(Sxx0), vmin=0, vmax=vmax, shading='gouraud')  # Remove vmin and vmax here
        if i == 0:
            plt.title("Placebo")
        plt.ylabel('f [Hz]')
        plt.ylim([ylim_min, ylim_max])

        ax = plt.subplot(len(p_indices)+1, 2, (2*i)+1+1)
        if event_start_line:
            plt.axvline(x = 0, color = 'w', linestyle='dashed',alpha=0.35)

        pcm = ax.pcolormesh(dt, f1, np.abs(Sxx1), vmin=0, vmax=vmax, shading='gouraud')  # Remove vmin and vmax here
        if i == 0:
            plt.title("Alcohol")
        # plt.ylabel('f [Hz]')
        plt.ylim([ylim_min, ylim_max])
        
        cbar_x = 0.95  # Adjust this value as needed to position the colorbar correctly
        cbar_width = 0.02
        cbar_height = 0.5 / len(p_indices)
        cbar_y = 0.33 + 0.65 * (i / len(p_indices))

        cbar_ax = plt.gcf().add_axes([cbar_x, cbar_y, cbar_width, cbar_height])
        plt.colorbar(pcm, cax=cbar_ax, label='Magnitude [dB]')
    plt.subplots_adjust(bottom=0.1, top=0.94)
    plt.suptitle(f"{channel} Channel")
    plt.show()
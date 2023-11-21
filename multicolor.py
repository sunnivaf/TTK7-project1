def multicolored_line(x,y,z):

# Create a set of line segments so that we can color them individually
# This creates the points as an N x 1 x 2 array so that we can stack points
# together easily to get the segments. The segments array for line collection
# needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)


    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(z.min(), z.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(z)
    lc.set_linewidth(2)
    return lc

def plot_multicolor(x,Y,Z, ax):
    # Create a set of line segments, where x is the x axis and y
    # is a list of the corresponding y-values at each x and Z
    # is a list of the corresponding z-values at each x. The color
    # of each line segment at each x-value is determined by Z.
    for i in range(len(Y)):
        ax.add_collection(multicolored_line(x, Y[i], Z[i]))
    ax.autoscale_view()

# Example
from scipy.signal import hilbert
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD

#data is 1D np array
# Perform hilbert transform on original data
duration = 4.0
Fs = 100.0
samples = int(Fs*duration)
t = np.arange(samples) / Fs

analytic_signal = hilbert(data)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                           (2.0*np.pi) * Fs)
# inst_freq = (np.diff(inst_phase) / (2.0*np.pi) * Fs)

inst_amp = np.abs(analytic_signal)



fig,ax = plt.subplots(2,1, figsize = (10,8), constrained_layout=True)
ylim = np.max(np.abs(instantaneous_frequency))

ax[0].plot(t, data, label = "Original Data")
ax[0].plot(t, inst_amp, label = "Instantaneous Amplitude")
ax[0].set_title("Signal and Instantaneous Amplitude")
ax[0].grid()

ax[1].set_facecolor('black')
ax[1].set_ylim([-ylim+0.1*ylim, ylim+0.1*ylim])
ax[1].set_xlim([0,4])
ax[1].set_title("Instantaneous Frequency")
ax[1].set_ylabel("Frequency (Hz)")
ax[1].set_xlabel("Time (s)")
ax[1].grid()
lc=multicolored_line(t[:-1],instantaneous_frequency,inst_amp[:-1])
line = ax[1].add_collection(lc)
fig.colorbar(line, ax=ax[1])
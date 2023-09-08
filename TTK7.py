from tftb.generators import amgauss, fmlin
import matplotlib.pyplot as plt
import numpy as np

# Define the time values
t = np.linspace(0, 3000, 1000)  # Time from 0 to 1 second, with 1000 points

# Define the frequencies of the three components
frequencies = [5, 12, 15]  # Frequencies in Hz

# Create the sinusoidal waveforms for all three components as a vector
components = np.sin(2 * np.pi * np.outer(frequencies, t))

# Combine the three components to create the signal
signal = np.sum(components, axis=0)

# Plot the signal
plt.figure(figsize=(10, 6))
plt.plot(t, signal)
plt.title('Signal with 5Hz, 12Hz, and 15Hz Components')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

## Analyse the final signal: stationary or non-stationary
# The final signal is stationary because it doesn't cahnge over time

# Run a FFT analysis to get an idea of the frequency components. 
# Reflect on the results of this analysis



# How to decide the window size if STFT or WT is going to be used?


# Which signal processing technique is best for your signal (FFT, STFT, WVT, WT, HT)?

# Add an offset and repeat the analysis

# Add white noise and repeat the analysis

# Add a linearly time varying frequency component and repeat the analysis (frequency=kt)

# Add an offset and white noise and repeat the analysis


"""Example of a Fourier transform to show how a series is converted from time to frequency"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq

# Generate time domain
FS = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 1, FS)

# Top time series: Simulated irregular waveform (sum of sine waves with random noise)
FREQS = [1, 4, 8, 16]  # Frequencies of components in Hz
signal1 = np.sin(2 * np.pi * FREQS[0] * t) - 1
signal2 = np.sin(2 * np.pi * FREQS[1] * t) * 0.5 + 0.5
signal3 = np.sin(2 * np.pi * FREQS[2] * t) * 0.25 + 1.25
signal4 = np.sin(2 * np.pi * FREQS[3] * t) * 0.125 + 1.625


# Bottom time series: Combination of three sine waves
composite_signal = signal1 + signal2 + signal3 + signal4

# Compute FFT
fft_signal1 = fft(signal1)
fft_signal2 = fft(signal2)
fft_signal3 = fft(signal3)
fft_signal4 = fft(signal4)
fft_composite_signal = fft(composite_signal)

# Frequency domain (x-axis for FFT)
freq = fftfreq(len(t), 1 / FS)

# Filter the positive frequencies for plotting
positive_freqs = freq > 0

# Plot the figures
plt.figure(figsize=(10, 8))

# Time series (Top)
plt.subplot(221)
plt.plot(t, signal1, color="blue")
plt.plot(t, signal2, color="green")
plt.plot(t, signal3, color="red")
plt.plot(t, signal4, color="orange")
plt.title("Time series")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [µV]")

# Spectrum (Top)
plt.subplot(222)
plt.plot(
    freq[positive_freqs],
    2.0 / len(t) * np.abs(fft_signal1[positive_freqs]),
    color="blue",
)
plt.plot(
    freq[positive_freqs],
    2.0 / len(t) * np.abs(fft_signal2[positive_freqs]),
    color="green",
)
plt.plot(
    freq[positive_freqs],
    2.0 / len(t) * np.abs(fft_signal3[positive_freqs]),
    color="red",
)
plt.plot(
    freq[positive_freqs],
    2.0 / len(t) * np.abs(fft_signal4[positive_freqs]),
    color="orange",
)
plt.xscale("log")
# plt.ylim(0, 0.3)
plt.title("Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [µV]")

# Time series (Bottom)
plt.subplot(223)
plt.plot(t, composite_signal, color="black")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [µV]")

# Spectrum (Bottom)
plt.subplot(224)
plt.plot(
    freq[positive_freqs],
    2.0 / len(t) * np.abs(fft_composite_signal[positive_freqs]),
    color="black",
)
plt.xscale("log")
# plt.ylim(0, 0.3)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [µV]")

plt.tight_layout()
plt.show()

"""
Smoothing of signals via wavelet reconstruction
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pywt

from simulation_consumption import consumption

# %%
# Generate values for i
i_values = np.linspace(1, 512, 1000)

# Calculate consumption values
consumption_values = consumption(i_values)

# %%
# Perform wavelet decomposition
WAVELET = "sym12"  ## Wavelet type, Symmlet 12
w = pywt.Wavelet(WAVELET)

## Choose the maximum decomposition level
levels = pywt.dwt_max_level(data_len=len(consumption_values), filter_len=w.dec_len)
print("Max decomposition level:", levels)

coeffs = pywt.wavedec(consumption_values, WAVELET, level=levels)

# %%
# ## Create dict for reconstructed signals
smooth_signals = {}

## Loop through levels and add detail level components
for l in range(levels):
    smooth_coeffs = coeffs.copy()
    smooth_signals[l] = {}
    ## Set remaining detail coefficients to zero
    for i in range(1, len(smooth_coeffs) - l):
        smooth_coeffs[i] = np.zeros_like(smooth_coeffs[i])
    smooth_signals[l]["coeffs"] = smooth_coeffs
    # Reconstruct the signal using only the approximation coefficients
    smooth_signals[l]["signal"] = pywt.waverec(smooth_coeffs, WAVELET)

# %%
plt.figure(figsize=(10, 6))
## Loop through levels and add detail level components
for i, (level, signal) in enumerate(smooth_signals.items(), 1):
    plt.subplot(len(smooth_signals), 1, i)  # Create a subplot for each smooth signal
    plt.plot(consumption_values, label="Original Signal")
    plt.plot(signal["signal"])
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(rf"Smooth Signal $S_{{j-{level}}}$")
    plt.legend()


plt.tight_layout()  # Adjust layout to prevent overlapping subplots
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Original Signal and Smooth Components")
plt.legend()
plt.show()

# %%

"""Example wavelet functions with their Fourier transform"""

import logging
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pywt  ## For Daubechies wavelet
from scipy.signal import morlet2, cwt
from scipy.fftpack import fft, fftshift
from scipy.special import gamma

from src.utils.logging_helpers import define_other_module_log_level

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("Error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Time domain
t = np.linspace(-4, 4, 1000)


# Define wavelet functions
def morlet_wavelet(time, w=5):
    """Morlet wavelet."""
    return np.cos(w * time) * np.exp(-(time**2) / 2)


def complex_morlet_wavelet(time, w=5):
    """Complex Morlet wavelet to avoid dual frequency peaks."""
    sigma = 1.0  # Standard deviation of the Gaussian envelope
    return np.exp(1j * w * time) * np.exp(-(time**2) / (2 * sigma**2))


def paul_wavelet(time, m=4):
    """Paul wavelet."""
    return (
        (2**m * 1j**m * gamma(m + 0.5))
        / (np.pi**0.5 * gamma(2 * m))
        * (1 - 1j * time) ** -(m + 1)
    )


def daubechies_wavelet(time, wavelet_function="db4", level=5):
    """Daubechies wavelet."""
    wavelet = pywt.Wavelet(wavelet_function)
    _, psi, _ = wavelet.wavefun(level=level)  # Get wavelet function
    return np.interp(
        time, np.linspace(-4, 4, len(psi)), psi
    )  # Interpolate for consistent t-domain


def dog_wavelet(time, m=6):
    """Corrected DOG wavelet (Derivative of Gaussian)."""
    # Compute the mth derivative of a Gaussian using recursion
    gaussian = np.exp(-(time**2) / 2)
    h_m = np.polyval(
        np.polyder([1] + [0] * m), time
    )  # Compute the mth Hermite polynomial
    return (-1) ** (m + 1) * h_m * gaussian


# Define Fourier transform function
def fourier_transform(wavelet):
    """Fourier transform on wavelets"""
    return fftshift(np.abs(fft(wavelet)))


# Create wavelets
wavelets = {
    # "Morlet": morlet_wavelet(t),
    "Complex Morlet": complex_morlet_wavelet(t),
    "Paul (m=4)": paul_wavelet(t, m=4),  # Take real part
    "Daubechies (db2)": daubechies_wavelet(t, "db2"),  # Daubechies wavelet
}

#  Plot
fig, axes = plt.subplots(len(wavelets), 2, figsize=(10, 10))

for i, (name, wavelet_func) in enumerate(wavelets.items()):
    # Plot wavelet in time domain
    axes[i, 0].plot(t, wavelet_func, color="black")
    axes[i, 0].set_ylabel(name, fontsize=14)
    axes[i, 0].set_xlabel("t / s")
    if i == 0:
        axes[i, 0].set_title("Time domain", fontsize=14)
    if i < len(wavelets) - 1:
        axes[i, 0].set_xlabel(None)

    # Fourier transform (frequency domain)
    freq = np.linspace(-2, 2, len(t))
    ft_wavelet = fourier_transform(wavelet_func)
    axes[i, 1].plot(freq, ft_wavelet, color="black")
    axes[i, 1].set_xlim([-0.5, 0.5])
    axes[i, 1].set_xlabel("s ω / (2π)")
    if i == 0:
        axes[i, 1].set_title("Frequency domain", fontsize=14)
    if i < len(wavelets) - 1:
        axes[i, 1].set_xlabel(None)

plt.tight_layout()

module_name = Path(__file__).name
module_name = module_name.split(".")[0]
logger.debug(module_name)
img = Path(__file__).parents[1] / "results" / f"{module_name}.png"
plt.savefig(img, bbox_inches="tight")

plt.show()

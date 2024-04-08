"""
Continuous wavelet transform of signals
"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet

from analysis import retrieve_data as rd

WAVELET_WIDTH = np.arange(1, 31)
WAVELET = morlet


def run_cwt(signal: list, wavelet: Callable, wavelet_length: int) -> tuple[int, list]:
    """Convert signal to wavelet coefficients"""
    ## Run continuous wavelet transform
    return cwt(signal, wavelet, wavelet_length)


def main() -> None:
    """Run script"""
    ## Get data
    raw_data = rd.get_insee_data("000857180")
    t, y = rd.clean_insee_data(raw_data)
    cwt_result = cwt(y, WAVELET, WAVELET_WIDTH)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the CWT result
    extent = [min(t), max(t), 1, 30]
    ax1.imshow(
        np.abs(cwt_result),
        extent=extent,
        cmap="jet",
        aspect="auto",
        vmax=np.abs(cwt_result).max(),
        vmin=0,
    )
    ax1.set_ylabel("Scale")
    ax1.set_title("Continuous Wavelet Transform")

    # Plot the time series data
    ax2.plot(t, y, color="blue")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Time Series")

    # # Add cone of influence (COI)
    # # The COI typically represents the regions where edge effects become significant
    # # You can customize the COI shading as desired
    # plt.fill_between([-1, 1], 1, 31, color="black")

    plt.show()


if __name__ == "__main__":
    main()

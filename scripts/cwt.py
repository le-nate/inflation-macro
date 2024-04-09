"""
Continuous wavelet transform of signals
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pywt

from analysis import retrieve_data as rd

WAVELET_WIDTH = np.arange(1, 64)  # np.geomspace(1, 16, num=4)
WAVELET = "morl"


def run_cwt(
    signal: list,
    wavelet_length: npt.NDArray,
    wavelet: Callable,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Convert signal to wavelet coefficients"""
    ## Run continuous wavelet transform
    cwtmatr, freqs = pywt.cwt(signal, wavelet_length, wavelet)
    return cwtmatr, freqs


def main() -> None:
    """Run script"""
    ## Get data
    raw_data = rd.get_fed_data("PCE", units="pc1")
    t, y = rd.clean_fed_data(raw_data)
    cwt_result, freqs = pywt.cwt(y, WAVELET_WIDTH, WAVELET)
    t_num = mdates.date2num(t)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the CWT result
    extent = [min(t_num), max(t_num), min(WAVELET_WIDTH), max(WAVELET_WIDTH)]
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

    # TODO Add cone of influence (COI)

    plt.show()


if __name__ == "__main__":
    main()

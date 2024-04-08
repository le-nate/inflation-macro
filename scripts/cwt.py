"""
Continuous wavelet transform of signals
"""

from typing import Callable

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pywt

from analysis import retrieve_data as rd

WAVELET_WIDTH = np.arange(1, 7)  # np.geomspace(1, 16, num=4)
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
    raw_data = rd.get_fed_data("CPIAUCSL", units="pc1", freq="m")
    t, y = rd.clean_fed_data(raw_data)
    cwt_result, frequencies = run_cwt(y, WAVELET_WIDTH, WAVELET)

    print(t)
    print(cwt_result)

    # Create subplots
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    # Plot the CWT result
    fig, axs = plt.subplots(2, 1, sharex=True)
    extent = [min(t), max(t), 1, 8]
    pcm = axs[0].pcolormesh(t, frequencies, cwt_result)
    axs[0].set_ylabel("Scale")
    axs[0].set_title("Continuous Wavelet Transform")
    fig.colorbar(pcm, ax=axs[0])

    # Plot the time series data
    axs[1].plot(t, y, color="blue")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Time Series")

    # # Add cone of influence (COI)
    # # The COI typically represents the regions where edge effects become significant
    # # You can customize the COI shading as desired
    # plt.fill_between([-1, 1], 1, 31, color="black")

    plt.show()


if __name__ == "__main__":
    main()

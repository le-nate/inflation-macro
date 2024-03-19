"""
Smoothing of signals via wavelet reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

from analysis import retrieve_data as rd


def trim_signal(signal: list) -> list:
    """Removes first or last observation for odd-numbered datasets"""
    ## Time series with uneven result in mismatched lengths with the reconstructed
    ## signal, so we remove a value from the approximated signal
    if len(signal) % 2 != 0:
        print(
            f"Odd number of observations dectected (Length: {len(signal)}). Trim data? (y/n)"
        )
        trim = input()
    else:
        trim = "n"

    if trim == "y":
        print("Trim beginning or end of time series? (begin/end)")
        trim = input()
    if trim == "begin":
        trimmed_signal = signal[1:]
    elif trim == "end":
        trimmed_signal = signal[:-1]
    else:
        trimmed_signal = signal
    return trimmed_signal


def smooth_signal(signal_coeffs: list, levels: int) -> dict:
    """Generate smoothed signals based off wavelet coefficients for each pre-defined level"""
    ## Initialize dict for reconstructed signals
    signals_dict = {}

    ## Loop through levels and add detail level components
    for l in range(levels):
        smooth_coeffs = signal_coeffs.copy()
        signals_dict[l] = {}
        ## Set remaining detail coefficients to zero
        for coeff in range(1, len(smooth_coeffs) - l):
            smooth_coeffs[coeff] = np.zeros_like(smooth_coeffs[coeff])
        signals_dict[l]["coeffs"] = smooth_coeffs
        # Reconstruct the signal using only the approximation coefficients
        reconst = pywt.waverec(smooth_coeffs, WAVELET)
        signals_dict[l]["signal"] = trim_signal(reconst)

    return signals_dict


if __name__ == "__main__":

    ## Matplotlib Settings
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    raw_data = rd.get_insee_data("000857179")
    t, y = rd.clean_insee_data(raw_data)

    ## Define the wavelet type
    WAVELET = "db4"
    w = pywt.Wavelet(WAVELET)

    ## Choose the maximum decomposition level
    levels = pywt.dwt_max_level(data_len=len(y), filter_len=w.dec_len)
    print(f"Max decomposition level of {levels} for time series length of {len(y)}")

    coeffs = pywt.wavedec(y, WAVELET, level=levels)

    smooth_signals = smooth_signal(coeffs, levels)

    ## Input name of time series
    print("Enter name of time series (to be included in plot)")
    name = input()

    fig = plt.figure(figsize=(10, 10))
    ## Loop through levels and add detail level components
    for i, (level, signal) in enumerate(smooth_signals.items(), 1):
        ## Subplot for each smooth signal
        plt.subplot(len(smooth_signals), 1, i)
        plt.plot(t, y, label=name.title())
        plt.plot(t, signal["signal"])
        plt.xlabel("Year")
        plt.grid()
        plt.title(rf"Approximation: $S_{{j-{level}}}$")
        plt.legend()

    plt.xlabel("Year")
    plt.ylabel(f"{name.capitalize()}")
    fig.suptitle(f"Wavelet smoothing of {name.lower()}")
    fig.tight_layout()
    plt.show()

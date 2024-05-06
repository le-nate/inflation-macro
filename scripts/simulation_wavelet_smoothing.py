"""Smoothing of signals via wavelet reconstruction with simulated data"""

# %%
import numpy as np
import matplotlib.pyplot as plt

from simulation_consumption import consumption
import scripts.dwt as dwt


def main() -> None:
    """Run script"""
    ## Matplotlib Settings
    small_size = 8
    medium_size = 10
    bigger_size = 12
    plt.rc("font", size=bigger_size)  # controls default text sizes
    plt.rc("axes", titlesize=bigger_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title

    i_values = np.linspace(1, 512, 1000)
    raw_data = consumption(i_values)
    t, y = i_values, raw_data

    ## Define the wavelet type
    wavelet_type = "sym12"
    smooth_signals = dwt.smooth_signal(y, wavelet_type)

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


if __name__ == "__main__":
    main()

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pycwt as wavelet
from pycwt.helpers import find

from analysis import retrieve_data as rd
from analysis import wavelet_transform as wt


def main() -> None:
    """Run script"""

    # * Load dataset
    measure_1 = "PSAVERT"
    raw_data = rd.get_fed_data(measure_1)
    df1, _, _ = rd.clean_fed_data(raw_data)

    measure_2 = "MICH"
    raw_data = rd.get_fed_data(measure_2)
    df2, _, _ = rd.clean_fed_data(raw_data)

    # * Pre-process data: Align time series temporally
    dfcombo = df1.merge(df2, how="left", on="date", suffixes=("_1", "_2"))
    print(dfcombo.info())
    dfcombo.dropna(inplace=True)
    print("\n", dfcombo.shape)

    # * Pre-process data: Standardize and detrend
    y1 = dfcombo["value_1"].to_numpy()
    y2 = dfcombo["value_2"].to_numpy()
    # # TODO check normalization approach
    # # p = np.polyfit(t - t0, dat, 1)
    # # dat_notrend = dat - np.polyval(p, t - t0)
    # std = y1.std()  #! dat_notrend.std()  # Standard deviation
    # var1 = std**2  # Variance
    # y1 = y1 / std  #! dat_notrend / std  # Normalized dataset
    # std = y2.std()  #! dat_notrend.std()  # Standard deviation
    # var2 = std**2  # Variance
    # y2 = y2 / std  #! dat_notrend / std  # Normalized dataset
    # # y1 = wt.standardize_data_for_xwt(y1, detrend=False, remove_mean=True)
    # # y2 = wt.standardize_data_for_xwt(y2, detrend=False, remove_mean=True)

    # *Prepare variables
    t = np.linspace(1, y1.size + 1, y1.size)
    print(t.shape, y1.size)
    dt = t[1] - t[0]
    dj = 1 / 12
    mother = wavelet.Morlet(6)  # Morlet wavelet with :math:`\omega_0=6`.

    # * Perform cross wavelet transform
    xwt_result, coi, freqs, signif = wavelet.xwt(y1, y2, dt=dt, dj=dj, wavelet=mother)

    # * Normalize results
    signal_size = y1.size
    period, power, sig95, coi_plot = wt.normalize_xwt_results(
        signal_size, xwt_result, coi, freqs, signif
    )

    # * Plot results
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot XWT
    extent = [min(t), max(t), min(coi_plot), max(period)]

    # Plot the time series data
    ax2.set_title("b) Time series")
    ax2.set_ylabel("Amplitude")
    ax2.plot(t, y1, "-", linewidth=1)
    ax2.plot(t, y2, "k", linewidth=1.5)

    # Normalized cwt power spectrum, signifance levels, and cone of influence
    # Period scale is logarithmic
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    ax1.contourf(
        t,
        np.log2(period),
        np.log2(power),
        np.log2(levels),
        extend="both",
        cmap="jet",
        extent=extent,
    )
    ax1.contour(
        t, np.log2(period), sig95, [-99, 1], colors="k", linewidths=2, extent=extent
    )
    ax1.fill(
        np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
        coi_plot,
        "k",
        alpha=0.3,
        hatch="x",
    )
    ax1.set_title("a) {} Cross-Wavelet Power Spectrum ({})".format("", mother.name))
    ax1.set_ylabel("Period (years)")
    #
    Yticks = 2 ** np.arange(
        np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))
    )
    ax1.set_yticks(np.log2(Yticks))
    ax1.set_yticklabels(Yticks)

    plt.show()


if __name__ == "__main__":
    main()

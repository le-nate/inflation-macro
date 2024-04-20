"""Cross wavelet transformation"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
import pandas as pd
import pycwt as wavelet
from pycwt.helpers import find

from analysis import retrieve_data as rd
from analysis import wavelet_transform as wt


def main() -> None:
    """Run script"""

    # * Load dataset
    measure_1 = "MICH"
    raw_data = rd.get_fed_data(measure_1)
    df1, _, _ = rd.clean_fed_data(raw_data)
    print(df1.head())
    print(df1.tail())

    measure_2 = "PCEND"
    raw_data = rd.get_fed_data(measure_2, units="pc1")
    df2, _, _ = rd.clean_fed_data(raw_data)
    print(df2.head())
    print(df2.tail())

    # measure_1 = "000857180"
    # raw_data = rd.get_insee_data(measure_1)
    # df1, _, _ = rd.clean_insee_data(raw_data)
    # print(df1.head())
    # print(df1.tail())

    # measure_2 = "000857181"
    # raw_data = rd.get_insee_data(measure_2)
    # df2, _, _ = rd.clean_insee_data(raw_data)
    # print(df2.head())
    # print(df2.tail())

    # * Pre-process data: Align time series temporally
    dfcombo = df1.merge(df2, how="left", on="date", suffixes=("_1", "_2"))
    dfcombo.dropna(inplace=True)

    # * Pre-process data: Standardize and detrend
    y1 = dfcombo["value_1"].to_numpy()
    y2 = dfcombo["value_2"].to_numpy()
    t = np.linspace(1, y1.size + 1, y1.size)
    # # # TODO check normalization approach
    # p = np.polyfit(t, y1, 1)
    # dat_notrend = y1 - np.polyval(p, t)
    # std = dat_notrend.std()  # Standard deviation
    # var1 = std**2  # Variance
    # y1 = dat_notrend / std  # Normalized dataset
    # std = y2.std()  #! dat_notrend.std()  # Standard deviation
    # var2 = std**2  # Variance
    # y2 = y2 / std  #! dat_notrend / std  # Normalized dataset
    y1 = wt.standardize_data_for_xwt(y1, detrend=False, remove_mean=True)
    y2 = wt.standardize_data_for_xwt(y2, detrend=False, remove_mean=True)

    # *Prepare variables
    dt = 1 / 12
    dj = 1 / 8
    s0 = 2 * dt
    mother = wavelet.Morlet(6)  # Morlet wavelet with :math:`\omega_0=6`.

    # * Perform cross wavelet transform
    xwt_result, coi, freqs, signif = wavelet.xwt(
        y1, y2, dt=dt, dj=dj, s0=s0, wavelet=mother, ignore_strong_trends=False
    )

    # * Caclulate wavelet coherence

    # Calculate the wavelet coherence (WTC). The WTC finds regions in time
    # frequency space where the two time seris co-vary, but do not necessarily have
    # high power.
    _, a_WCT, _, _, _ = wavelet.wct(
        y1,
        y2,
        dt,
        dj=1 / 12,
        s0=-1,
        J=-1,
        sig=False,  #! To save time
        # significance_level=0.8646,
        wavelet="morlet",
        normalize=True,
        cache=True,
    )
    # * Calculate phase
    # Calculates the phase between both time series. The phase arrows in the
    # cross wavelet power spectrum rotate clockwise with 'north' origin.
    # The relative phase relationship convention is the same as adopted
    # by Torrence and Webster (1999), where in phase signals point
    # upwards (N), anti-phase signals point downwards (S). If X leads Y,
    # arrows point to the right (E) and if X lags Y, arrow points to the
    # left (W).
    angle = 0.5 * np.pi - a_WCT
    u, v = np.cos(angle), np.sin(angle)
    print(len(u), len(v))

    # * Normalize results
    signal_size = y1.size
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    period, power, sig95, coi_plot = wt.normalize_xwt_results(
        signal_size, xwt_result, coi, np.log2(levels[2]), freqs, signif
    )

    # * Plot results
    print(dfcombo.head())
    print(dfcombo.info())
    # Create subplots
    fig, (ax) = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # Plot XWT
    extent = [min(t), max(t), min(coi_plot), max(period)]

    # Normalized cwt power spectrum
    ax.contourf(
        t,
        np.log2(period),
        np.log2(power),
        np.log2(levels),
        extend="both",
        cmap="jet",
        extent=extent,
    )

    # Plot signifance levels
    ax.contour(
        t, np.log2(period), sig95, [-99, 1], colors="k", linewidths=2, extent=extent
    )
    # Plot coi
    ax.fill(
        np.concatenate([t, t[-1:] + dt, t[-1:] + dt, t[:1] - dt, t[:1] - dt]),
        coi_plot,
        "k",
        alpha=0.3,
        hatch="--",
    )
    print(
        t[::12],
        np.log2(period[::8]),
        u[::12, ::12],
        v[::12, ::12],
    )
    # * Plot phase difference arrows
    ax.quiver(
        t[::12],
        np.log2(period[::8]),
        u[::12, ::12],
        v[::12, ::12],
        units="width",
        angles="uv",
        pivot="mid",
        linewidth=1,
        edgecolor="k",
        headwidth=2,
        headlength=2,
        headaxislength=1,
        minshaft=0.2,
        minlength=0.5,
    )

    ax.set_ylim(ax.get_ylim()[::-1])

    # # TODO Plot phase difference as time series
    # # ! Not correct
    # ax3.plot(t, phase_diff, "y", label="Phase difference")
    # ax3.plot(t, np.angle(y1), "b")
    # ax3.plot(t, np.angle(y2), "r")

    ax.set_title(f"a) Cross-Wavelet Power Spectrum ({measure_1} X {measure_2})")
    ax.set_ylabel("Period (months)")
    #
    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))
    )
    ax.set_yticks(np.log2(y_ticks))
    ax.set_yticklabels(y_ticks)

    plt.show()


if __name__ == "__main__":
    main()

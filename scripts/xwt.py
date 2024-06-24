"""Cross wavelet transformation"""

from __future__ import division
import logging
import sys
from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find

from src.helpers import define_other_module_log_level
from src import retrieve_data
from src import wavelet_helpers

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Constants
DT = 1 / 12  # Delta t
DJ = 1 / 8  # Delta j
S0 = 2 * DT  # Initial scale
MOTHER = "morlet"  # Morlet wavelet with :math:`\omega_0=6`.
MOTHER_DICT = {
    "morlet": wavelet.Morlet(6),
    "paul": wavelet.Paul(),
    "DOG": wavelet.DOG(),
    "mexicanhat": wavelet.MexicanHat(),
}
LEVELS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]


@dataclass
class DataForXWT(object):
    """Holds data for XWT"""

    def __init__(
        self,
        y1_values: npt.NDArray,
        y2_values: npt.NDArray,
        mother_wavelet: Type,
        delta_t: float,
        delta_j: float,
        initial_scale: float,
        levels: List[float],
    ) -> None:
        self.y1_values = y1_values
        self.y2_values = y2_values
        self.mother_wavelet = mother_wavelet
        self.delta_t = delta_t
        self.delta_j = delta_j
        self.initial_scale = initial_scale
        self.levels = levels


@dataclass
class ResultsFromXWT(object):
    """Holds results from Cross-Wavelet Transform"""

    def __init__(
        self,
        power: npt.NDArray,
        period: npt.NDArray,
        significance_levels: npt.NDArray,
        coi: npt.NDArray,
        phase_diff_u: npt.NDArray,
        phase_diff_v: npt.NDArray,
    ) -> None:
        self.power = power
        self.period = period
        self.significance_levels = significance_levels
        self.coi = coi
        self.phase_diff_u = phase_diff_u
        self.phase_diff_v = phase_diff_v


def run_xwt(
    cross_wavelet_transform: Type[DataForXWT],
    ignore_strong_trends: bool = False,
    normalize: bool = True,
) -> Type[ResultsFromXWT]:
    """Conduct Cross-Wavelet Transformation on two series.\n
    Returns cross-wavelet power, period, significance levels, cone of influence,
    and phase"""

    # * Perform cross wavelet transform
    xwt_result, coi, freqs, signif = wavelet.xwt(
        y1=cross_wavelet_transform.y1_values,
        y2=cross_wavelet_transform.y2_values,
        dt=cross_wavelet_transform.delta_t,
        dj=cross_wavelet_transform.delta_j,
        s0=cross_wavelet_transform.initial_scale,
        wavelet=cross_wavelet_transform.mother_wavelet,
        ignore_strong_trends=ignore_strong_trends,
    )

    if normalize:
        # * Normalize results
        signal_size = cross_wavelet_transform.y1_values.size
        period, power, sig95, coi_plot = wavelet_helpers.normalize_xwt_results(
            signal_size,
            xwt_result,
            coi,
            np.log2(cross_wavelet_transform.levels[2]),
            freqs,
            signif,
        )
    else:
        period = 1 / freqs
        power = xwt_result
        sig95 = np.ones([1, signal_size]) * signif[:, None]
        sig95 = power / sig95  ## Want where power / sig95 > 1
        coi_plot = coi

    # * Caclulate wavelet coherence
    mother_wavelet_for_coherence = cross_wavelet_transform.mother_wavelet
    logger.debug(
        "Using mother wavelet *%s* for coherence", mother_wavelet_for_coherence
    )
    _, phase, _, _, _ = wavelet.wct(
        cross_wavelet_transform.y1_values,
        cross_wavelet_transform.y2_values,
        cross_wavelet_transform.delta_t,
        delta_j=cross_wavelet_transform.delta_t,
        s0=-1,
        J=-1,
        sig=False,  #! To save time
        # significance_level=0.8646,
        wavelet=mother_wavelet_for_coherence,
        normalize=True,
        cache=True,
    )

    # * Calculate phase difference
    phase_diff_u, phase_diff_v = calculate_phase_difference(phase)

    return ResultsFromXWT(power, period, sig95, coi_plot, phase_diff_u, phase_diff_v)


def calculate_phase_difference(
    xwt_phase: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Calculates the phase between both time series.\n
    Returns u and v phase difference vectors \n
    Description:\n
    These phase arrows follow the convention of Torrence and Webster (1999).
    In-phase points north, and anti-phase point south. If y1 leads y2, the arrows
    point east, and if y2 leads y1, the arrows point west."""

    angle = 0.5 * np.pi - xwt_phase
    xwt_u, xwt_v = np.cos(angle), np.sin(angle)
    logger.debug("Comparing length of phase arrays: %s, %s", len(xwt_u), len(xwt_v))
    return xwt_u, xwt_v


def main() -> None:
    """Run script"""

    # * Load dataset
    measure_1 = "MICH"
    raw_data = retrieve_data.get_fed_data(measure_1)
    df1, _, _ = retrieve_data.clean_fed_data(raw_data)

    measure_2 = "PCEDG"
    raw_data = retrieve_data.get_fed_data(measure_2, units="pc1")
    df2, _, _ = retrieve_data.clean_fed_data(raw_data)

    # measure_1 = "000857180"
    # raw_data = retrieve_data.get_insee_data(measure_1)
    # df1, _, _ = retrieve_data.clean_insee_data(raw_data)

    # measure_2 = "000857181"
    # raw_data = retrieve_data.get_insee_data(measure_2)
    # df2, _, _ = retrieve_data.clean_insee_data(raw_data)

    # * Pre-process data: Align time series temporally
    dfcombo = df1.merge(df2, how="left", on="date", suffixes=("_1", "_2"))
    dfcombo.dropna(inplace=True)

    # * Pre-process data: Standardize and detrend
    y1 = dfcombo["value_1"].to_numpy()
    y2 = dfcombo["value_2"].to_numpy()
    t = np.linspace(1, y1.size + 1, y1.size)
    y1 = wavelet_helpers.standardize_data_for_xwt(y1, detrend=False, remove_mean=True)
    y2 = wavelet_helpers.standardize_data_for_xwt(y2, detrend=False, remove_mean=True)

    mother_xwt = MOTHER_DICT[MOTHER]

    xwt_data = DataForXWT(
        y1,
        y2,
        mother_xwt,
        DT,
        DJ,
        S0,
        LEVELS,
    )

    results_from_xwt = run_xwt(xwt_data, ignore_strong_trends=False)

    # # *Prepare variables
    # dt = 1 / 12
    # dj = 1 / 8
    # s0 = 2 * dt
    # mother = wavelet.Morlet(6)  # Morlet wavelet with :math:`\omega_0=6`.

    # # * Perform cross wavelet transform
    # xwt_result, coi, freqs, signif = wavelet.xwt(
    #     y1, y2, dt=dt, dj=dj, s0=s0, wavelet=mother, ignore_strong_trends=False
    # )

    # # * Normalize results
    # signal_size = y1.size
    # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    # period, power, sig95, coi_plot = wavelet_helpers.normalize_xwt_results(
    #     signal_size, xwt_result, coi, np.log2(levels[2]), freqs, signif
    # )

    # # * Caclulate wavelet coherence
    # _, phase, _, _, _ = wavelet.wct(
    #     y1,
    #     y2,
    #     dt,
    #     dj=1 / 12,
    #     s0=-1,
    #     J=-1,
    #     sig=False,  #! To save time
    #     # significance_level=0.8646,
    #     wavelet="morlet",
    #     normalize=True,
    #     cache=True,
    # )

    # # * Calculate phase
    # # Calculates the phase between both time series. The phase arrows in the
    # # cross wavelet power spectrum rotate clockwise with 'north' origin.
    # # The relative phase relationship convention is the same as adopted
    # # by Torrence and Webster (1999), where in phase signals point
    # # upwards (N), anti-phase signals point downwards (S). If X leads Y,
    # # arrows point to the right (E) and if X lags Y, arrow points to the
    # # left (W).
    # angle = 0.5 * np.pi - phase
    # u, v = np.cos(angle), np.sin(angle)
    logger.debug(
        "Comparing length of phase arrays: %s, %s",
        len(results_from_xwt.phase_diff_u),
        len(results_from_xwt.phase_diff_v),
    )

    # * Plot results
    print(dfcombo.head())
    print(dfcombo.info())

    # * Plot XWT power spectrum
    fig, (ax) = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # Plot XWT
    extent = [
        min(t),
        max(t),
        min(results_from_xwt.coi),
        max(results_from_xwt.period),
    ]

    # Normalized cwt power spectrum
    ax.contourf(
        t,
        np.log2(results_from_xwt.period),
        np.log2(results_from_xwt.power),
        np.log2(xwt_data.levels),
        extend="both",
        cmap="jet",
        extent=extent,
    )

    # Plot signifance levels
    ax.contour(
        t,
        np.log2(results_from_xwt.period),
        results_from_xwt.significance_levels,
        [-99, 1],
        colors="k",
        linewidths=2,
        extent=extent,
    )
    # Plot coi
    ax.fill(
        np.concatenate(
            [
                t,
                t[-1:] + xwt_data.delta_t,
                t[-1:] + xwt_data.delta_t,
                t[:1] - xwt_data.delta_t,
                t[:1] - xwt_data.delta_t,
            ]
        ),
        results_from_xwt.coi,
        "k",
        alpha=0.3,
        hatch="--",
    )
    print(
        t[::12].shape,
        np.log2(results_from_xwt.period[::8]).shape,
        results_from_xwt.phase_diff_u[::12, ::12].shape,
        results_from_xwt.phase_diff_v[::12, ::12].shape,
    )
    # * Plot phase difference arrows
    ax.quiver(
        t[::12],
        np.log2(results_from_xwt.period[::8]),
        results_from_xwt.phase_diff_u[::12, ::12],
        results_from_xwt.phase_diff_v[::12, ::12],
        units="width",
        angles="uv",
        pivot="mid",
        linewidth=0.5,
        edgecolor="k",
        alpha=0.7,
        # headwidth=2,
        # headlength=2,
        # headaxislength=1,
        # minshaft=0.2,
        # minlength=0.5,
    )

    # * Invert y axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # # * Set x axis tick labels
    # start = dfcombo["date"].dt.year.iat[0]
    # end = dfcombo["date"].dt.year.iat[len(dfcombo) - 1]
    # print(start, end)
    # x_ticks = np.arange(start, end, 6)
    # # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(x_ticks)

    # * Set y axis tick labels
    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(results_from_xwt.period.min())),
        np.ceil(np.log2(results_from_xwt.period.max())),
    )
    ax.set_yticks(np.log2(y_ticks))
    ax.set_yticklabels(y_ticks)

    ax.set_title("Inflation Expectations X Durables Consumption (US)")
    ax.set_ylabel("Period (years)")

    plt.show()


if __name__ == "__main__":
    main()

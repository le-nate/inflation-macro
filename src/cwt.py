"""
Continuous wavelet transform of signals
based off: https://pycwt.reaDThedocs.io/en/latest/tutorial.html
"""

from __future__ import division
import logging
import sys
from dataclasses import dataclass

from typing import List, Tuple, Type

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import pycwt as wavelet

from src.helpers import define_other_module_log_level
from src import retrieve_data
from src.wavelet_helpers import plot_cone_of_influence, plot_signficance_levels

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Define title and labels for plots
LABEL = "Expected Inflation (US)"
UNITS = "%"

MEASURE = "MICH"

NORMALIZE = True  # Define normalization
DT = 1 / 12  # In years
S0 = 2 * DT  # Starting scale
DJ = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / DJ  # Seven powers of two with DJ sub-octaves
MOTHER = wavelet.Morlet(f0=6)
LEVELS = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]  # Period scale is logarithmic


@dataclass
class DataForCWT:
    """Holds data for continuous wavelet transform"""

    def __init__(
        self,
        t_values: npt.NDArray,
        y_values: npt.NDArray,
        mother_wavelet: Type,
        delta_t: float,
        delta_j: float,
        initial_scale: float,
        levels: List[float],
    ) -> None:
        self.t_values = t_values
        self.y_values = y_values
        self.mother_wavelet = mother_wavelet
        self.delta_t = delta_t
        self.delta_j = delta_j
        self.initial_scale = initial_scale
        self.levels = levels
        self.time_range()

    def time_range(self) -> npt.NDArray:
        """Takes first date and creates array with date based on defined dt"""
        # Define starting time and time step
        t0 = min(self.t_values)
        logger.debug("t0 type %s", type(t0))
        t0 = t0.astype("datetime64[Y]").astype(int) + 1970
        num_observations = self.t_values.size
        self.time_range = np.arange(1, num_observations + 1) * self.delta_t + t0
        return np.arange(1, num_observations + 1) * self.delta_t + t0


@dataclass
class ResultsFromCWT:
    """Holds results from continuous wavelet transform"""

    def __init__(
        self,
        power: npt.NDArray,
        period: npt.NDArray,
        significance_levels: npt.NDArray,
        coi: npt.NDArray,
    ) -> None:
        self.power = power
        self.period = period
        self.significance_levels = significance_levels
        self.coi = coi


# * Functions
def run_cwt(
    cwt_data: Type[DataForCWT],
    normalize: bool = True,
) -> Type[ResultsFromCWT]:
    """Conducts Continuous Wavelet Transform\n
    Returns power spectrum, period, cone of influence, and significance levels (95%)"""
    # p = np.polyfit(t - t0, dat, 1)
    # dat_notrend = dat - np.polyval(p, t - t0)
    std = cwt_data.y_values.std()  #! dat_notrend.std()  # Standard deviation

    if normalize:
        dat_norm = cwt_data.y_values / std  #! dat_notrend / std  # Normalized dataset
    else:
        dat_norm = cwt_data.y_values

    alpha, _, _ = wavelet.ar1(cwt_data.y_values)  # Lag-1 autocorrelation for red noise

    # * Conduct transformations
    # Wavelet transform
    wave, scales, freqs, cwt_coi, _, _ = wavelet.cwt(
        dat_norm, DT, DJ, S0, J, cwt_data.mother_wavelet
    )
    # Normalized wavelet power spectrum
    cwt_power = (np.abs(wave)) ** 2
    # Normalized Fourier equivalent periods
    cwt_period = 1 / freqs

    # * Statistical significance
    # where the ratio ``cwt_power / sig95 > 1``.
    num_observations = len(cwt_data.t_values)
    signif, _ = wavelet.significance(
        1.0,
        DT,
        scales,
        0,
        alpha,
        significance_level=0.95,
        wavelet=cwt_data.mother_wavelet,
    )
    cwt_sig95 = np.ones([1, num_observations]) * signif[:, None]
    cwt_sig95 = cwt_power / cwt_sig95

    return ResultsFromCWT(cwt_power, cwt_period, cwt_sig95, cwt_coi)


def plot_cwt(
    cwt_ax: plt.Axes,
    cwt_data: Type[DataForCWT],
    cwt_results: Type[ResultsFromCWT],
    include_significance: bool = True,
    include_cone_of_influence: bool = True,
    **kwargs
) -> None:
    """Plot Power Spectrum for Continuous Wavelet Transform"""
    _ = cwt_ax.contourf(
        cwt_data.time_range,
        np.log2(cwt_results.period),
        np.log2(cwt_results.power),
        np.log2(cwt_data.levels),
        extend="both",
        cmap=kwargs["cmap"],
    )

    if include_significance:
        plot_signficance_levels(
            cwt_ax,
            cwt_results.significance_levels,
            cwt_data.time_range,
            cwt_results.period,
            **kwargs,
        )

    if include_cone_of_influence:
        plot_cone_of_influence(
            cwt_ax,
            cwt_results.coi,
            cwt_data.time_range,
            cwt_data.levels,
            cwt_results.period,
            cwt_data.delta_t,
            tranform_type="cwt",
            **kwargs,
        )

    # * Invert y axis
    cwt_ax.set_ylim(cwt_ax.get_ylim()[::-1])

    y_ticks = 2 ** np.arange(
        np.ceil(np.log2(cwt_results.period.min())),
        np.ceil(np.log2(cwt_results.period.max())),
    )
    cwt_ax.set_yticks(np.log2(y_ticks))
    cwt_ax.set_yticklabels(y_ticks)


def main() -> None:
    """Run script"""
    # * Retrieve dataset
    raw_data = retrieve_data.get_fed_data(MEASURE, units="pc1", freqs="m")
    _, t_date, y = retrieve_data.clean_fed_data(raw_data)

    data_for_cwt = DataForCWT(t_date, y, MOTHER, DT, DJ, S0, LEVELS)

    results_from_cwt = run_cwt(data_for_cwt, normalize=True)

    # * Plot results
    plt.close("all")
    # plt.ioff()
    figprops = {"figsize": (20, 10), "dpi": 72}
    _, ax = plt.subplots(1, 1, **figprops)

    # * Add plot features
    cwt_plot_props = {
        "cmap": "jet",
        "sig_colors": "k",
        "sig_linewidths": 2,
        "coi_color": "k",
        "coi_alpha": 0.3,
        "coi_hatch": "--",
    }
    plot_cwt(ax, data_for_cwt, results_from_cwt, data_for_cwt, **cwt_plot_props)

    # * Set labels/title
    ax.set_xlabel("")
    ax.set_ylabel("Period (years)")
    ax.set_title(LABEL)

    plt.show()


if __name__ == "__main__":
    main()

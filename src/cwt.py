"""
Continuous wavelet transform of signals
based off: https://pycwt.reaDThedocs.io/en/latest/tutorial.html
"""

from __future__ import division
import logging
from pathlib import Path
import sys
from dataclasses import dataclass, field

from typing import List, Type

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import pycwt as wavelet

from constants import ids
from src.utils.logging_helpers import define_other_module_log_level
from src import retrieve_data
from src.utils.wavelet_helpers import (
    plot_cone_of_influence,
    plot_signficance_levels,
    standardize_series,
)

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Define title and labels for plots
LABEL = "Expected inflation"
UNITS = "%"

MEASURE = ids.US_INF_EXPECTATIONS

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

    t_values: npt.NDArray
    y_values: npt.NDArray
    mother_wavelet: Type
    delta_t: float
    delta_j: float
    initial_scale: float
    levels: List[float]
    time_range: npt.NDArray = field(init=False)

    def __post_init__(self):
        self.time_range = self.time_range(self)

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

    power: npt.NDArray
    period: npt.NDArray
    significance_levels: npt.NDArray
    coi: npt.NDArray


# * Functions
def run_cwt(
    cwt_data: Type[DataForCWT],
    normalize: bool = True,
    standardize: bool = False,
    **kwargs,
) -> Type[ResultsFromCWT]:
    """Conducts Continuous Wavelet Transform\n
    Returns power spectrum, period, cone of influence, and significance levels (95%)"""
    # p = np.polyfit(t - t0, dat, 1)
    # dat_notrend = dat - np.polyval(p, t - t0)
    std = cwt_data.y_values.std()  #! dat_notrend.std()  # Standard deviation

    if normalize:
        dat_norm = cwt_data.y_values / std  #! dat_notrend / std  # Normalized dataset
    if standardize:
        dat_norm = standardize_series(cwt_data.y_values, **kwargs)
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
    **kwargs,
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
    cwt_ax.set_yticklabels(y_ticks, size=15)


def main() -> None:
    """Run script"""
    # * Retrieve dataset
    raw_data = retrieve_data.get_fed_data(MEASURE)
    df, t_date, _ = retrieve_data.clean_fed_data(raw_data)
    df.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

    data_for_cwt = DataForCWT(
        t_date,
        df[f"{ids.EXPECTATIONS}"].to_numpy(),
        MOTHER,
        DT,
        DJ,
        S0,
        LEVELS,
    )

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
    plot_cwt(ax, data_for_cwt, results_from_cwt, **cwt_plot_props)

    # * Set labels/title
    ax.set_xlabel("", size=20)
    ax.set_ylabel("Period (years)", size=20)
    ax.set_title(LABEL, size=20)

    # * Export plot
    parent_dir = Path(__file__).parents[1]
    export_file = parent_dir / "results" / f"CWT_{LABEL}.png"
    plt.savefig(export_file, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()

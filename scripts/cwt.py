"""
Continuous wavelet transform of signals
based off: https://pycwt.reaDThedocs.io/en/latest/tutorial.html
"""

from __future__ import division
import logging
import sys

from typing import Dict, Generator, List, Tuple, Type, Union
from datetime import datetime
import numpy as np
import numpy.typing as npt
import matplotlib.figure
import matplotlib.pyplot as plt

import pycwt as wavelet
from pycwt.helpers import find

from analysis.helpers import define_other_module_log_level
from analysis import retrieve_data as rd

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


# * Functions
def set_time_range(t_array: npt.NDArray, dt: float) -> Tuple[float, npt.NDArray]:
    """Takes first date and creates array with date based on defined dt"""
    # Define starting time and time step
    t0 = min(t_array)
    logger.debug("t0 type %s", type(t0))
    t0 = t0.astype("datetime64[Y]").astype(int) + 1970
    num_observations = t_array.size
    return num_observations, np.arange(1, num_observations + 1) * dt + t0


def run_cwt(
    t_values: npt.NDArray,
    y_values: npt.NDArray,
    mother_wavelet: Type,
    normalize: bool = True,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Conducts Continuous Wavelet Transform"""
    # p = np.polyfit(t - t0, dat, 1)
    # dat_notrend = dat - np.polyval(p, t - t0)
    std = y_values.std()  #! dat_notrend.std()  # Standard deviation

    if normalize:
        dat_norm = y_values / std  #! dat_notrend / std  # Normalized dataset
    else:
        dat_norm = y_values

    alpha, _, _ = wavelet.ar1(y_values)  # Lag-1 autocorrelation for red noise

    # * Conduct transformations
    # Wavelet transform
    wave, scales, freqs, cwt_coi, _, _ = wavelet.cwt(
        dat_norm, DT, DJ, S0, J, mother_wavelet
    )
    # Normalized wavelet power spectrum
    cwt_power = (np.abs(wave)) ** 2
    # Normalized Fourier equivalent periods
    cwt_period = 1 / freqs

    # * Statistical significance
    # where the ratio ``cwt_power / sig95 > 1``.
    num_observations = len(t_values)
    signif, _ = wavelet.significance(
        1.0, DT, scales, 0, alpha, significance_level=0.95, wavelet=mother_wavelet
    )
    cwt_sig95 = np.ones([1, num_observations]) * signif[:, None]
    cwt_sig95 = cwt_power / cwt_sig95

    return cwt_power, cwt_period, cwt_coi, cwt_sig95


def plot_signficance_levels(
    cwt_ax: plt.Axes,
    signficance_levels: npt.NDArray,
    t_values: npt.NDArray,
    cwt_period: npt.NDArray,
    **kwargs
) -> None:
    """Plot contours for 95% significance level"""
    extent = [t_values.min(), t_values.max(), 0, max(cwt_period)]
    cwt_ax.contour(
        t_values,
        np.log2(cwt_period),
        signficance_levels,
        [-99, 1],
        colors=kwargs["colors"],
        linewidths=kwargs["linewidths"],
        extent=extent,
    )


def plot_cone_of_influence(
    cwt_ax: plt.Axes, cwt_coi: npt.NDArray, t_values, levels, cwt_period, **kwargs
) -> None:
    """Plot shaded area for cone of influence, where edge effects may occur"""
    alpha = kwargs["alpha"]
    hatch = kwargs["hatch"]
    cwt_ax.fill(
        np.concatenate(
            [
                t_values,
                t_values[-1:] + DT,
                t_values[-1:] + DT,
                t_values[:1] - DT,
                t_values[:1] - DT,
            ]
        ),
        np.concatenate(
            [
                np.log2(cwt_coi),
                [levels[2]],
                np.log2(cwt_period[-1:]),
                np.log2(cwt_period[-1:]),
                [levels[2]],
            ]
        ).clip(
            min=-2.5
        ),  # ! To keep cone of influence from bleeding off graph
        "k",
        alpha=alpha,
        hatch=hatch,
    )


def plot_cwt(
    cwt_ax: plt.Axes,
    t_values: npt.NDArray,
    cwt_power: npt.NDArray,
    cwt_period: npt.NDArray,
    levels: List[float],
    include_significance: bool = True,
    include_cone_of_influence: bool = True,
    **kwargs
) -> plt.Axes:
    """Plot Power Spectrum for Continuous Wavelet Transform"""
    power_spec = cwt_ax.contourf(
        t_values,
        np.log2(cwt_period),
        np.log2(cwt_power),
        np.log2(levels),
        extend="both",
        cmap="jet",
    )

    if include_significance:
        plot_signficance_levels(
            cwt_ax, kwargs["cwt_sig95"], t_values, cwt_period, **kwargs
        )

    if include_cone_of_influence:
        plot_cone_of_influence(
            cwt_ax=cwt_ax,
            t_values=t_values,
            cwt_period=cwt_period,
            levels=levels,
            cwt_coi=kwargs["cwt_coi"],
            alpha=kwargs["alpha"],
            hatch=kwargs["hatch"],
        )

    # * Invert y axis
    cwt_ax.set_ylim(cwt_ax.get_ylim()[::-1])

    Yticks = 2 ** np.arange(
        np.ceil(np.log2(cwt_period.min())), np.ceil(np.log2(cwt_period.max()))
    )
    cwt_ax.set_yticks(np.log2(Yticks))
    cwt_ax.set_yticklabels(Yticks)
    return cwt_ax


def main() -> None:
    """Run script"""
    # * Retrieve dataset
    raw_data = rd.get_fed_data(MEASURE, units="pc1", freqs="m")
    _, t_date, y = rd.clean_fed_data(raw_data)

    _, t = set_time_range(t_date, DT)

    power, period, coi, sig95 = run_cwt(t, y, mother_wavelet=MOTHER, normalize=True)

    # * Plot results
    plt.close("all")
    # plt.ioff()
    figprops = {"figsize": (20, 10), "dpi": 72}
    fig, ax = plt.subplots(1, 1, **figprops)

    cwt_plot_props = {
        "cwt_sig95": sig95,
        "cwt_coi": coi,
        "colors": "k",
        "linewidths": 3,
        "alpha": 0.3,
        "hatch": "--",
    }
    power_spec = plot_cwt(ax, t, power, period, LEVELS, **cwt_plot_props)

    # * Set labels/title
    ax.set_xlabel("")
    ax.set_ylabel("Period (years)")
    ax.set_title(LABEL)

    plt.show()


if __name__ == "__main__":
    main()

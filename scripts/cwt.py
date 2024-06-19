"""
Continuous wavelet transform of signals
based off: https://pycwt.reaDThedocs.io/en/latest/tutorial.html
"""

from __future__ import division
import logging
import sys

from typing import Dict, Generator, List, Tuple, Union
from datetime import datetime
import numpy as np
import numpy.typing as npt
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

NORMALIZE = True  # ! Define normalization
MEASURE = "MICH"
DT = 1 / 12  # In years


# * Functions
def set_time_range(t_array: npt.NDArray, dt: float) -> Tuple[float, npt.NDArray]:
    """Takes first date and creates array with date based on defined dt"""
    # Define starting time and time step
    t0 = min(t_array)
    logger.debug("t0 type %s", type(t0))
    t0 = t0.astype("datetime64[Y]").astype(int) + 1970
    num_observations = t_array.size
    return num_observations, np.arange(1, num_observations + 1) * dt + t0


def main() -> None:
    """Run script"""
    # * Retrieve dataset
    raw_data = rd.get_fed_data(MEASURE, units="pc1", freqs="m")
    _, t_date, dat = rd.clean_fed_data(raw_data)
    print("t_date type %s", type(t_date))

    # TODO check normalization approach
    # p = np.polyfit(t - t0, dat, 1)
    # dat_notrend = dat - np.polyval(p, t - t0)
    std = dat.std()  #! dat_notrend.std()  # Standard deviation
    var = std**2  # Variance

    if NORMALIZE:
        dat_norm = dat / std  #! dat_notrend / std  # Normalized dataset
    else:
        dat_norm = dat

    # * Define wavelet parameters
    mother = wavelet.Morlet(6)  # Morlet wavelet with :math:`\omega_0=6`.
    S0 = 2 * DT  # Starting scale
    DJ = 1 / 12  # Twelve sub-octaves per octaves
    J = 7 / DJ  # Seven powers of two with DJ sub-octaves
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # * Conduct ransformations
    # Wavelet transform
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        dat_norm, DT, DJ, S0, J, mother
    )
    # Inverse wavelet transform
    iwave = wavelet.icwt(wave, scales, DT, DJ, mother) * std
    # Normalized wavelet power spectrum
    power = (np.abs(wave)) ** 2
    # Normalized Fourier power spectrum
    fft_power = np.abs(fft) ** 2
    # Normalized Fourier equivalent periods
    period = 1 / freqs

    # * Statistical significance
    # where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(
        1.0, DT, scales, 0, alpha, significance_level=0.95, wavelet=mother
    )
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # * Calculate global wavelet and significance level
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(
        var, DT, scales, 1, alpha, significance_level=0.95, dof=dof, wavelet=mother
    )

    # * Calculate the scale average between 2 years and 8 years and significance level
    sel = find((period >= 2) & (period < 8))
    Cdelta = mother.cdelta
    scale_avg = (scales * np.ones((N, 1))).transpose()
    scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
    scale_avg = var * DJ * DT / Cdelta * scale_avg[sel, :].sum(axis=0)
    scale_avg_signif, tmp = wavelet.significance(
        var,
        DT,
        scales,
        2,
        alpha,
        significance_level=0.95,
        dof=[scales[sel[0]], scales[sel[-1]]],
        wavelet=mother,
    )

    # * Plot results
    plt.close("all")
    # plt.ioff()
    figprops = {"figsize": (20, 10), "dpi": 72}
    fig, bx = plt.subplots(1, 1, **figprops)

    # # 1) original series anomaly and the inverse wavelet transform
    # ax = plt.axes([0.1, 0.75, 0.65, 0.2])
    # ax.plot(t, iwave, "-", linewidth=1, color=[0.5, 0.5, 0.5])
    # ax.plot(t, dat, "k", linewidth=1.5)
    # ax.set_title("a) {}".format(TITLE))
    # ax.set_ylabel(r"{} [{}]".format(LABEL, UNITS))

    # * Normalized cwt power spectrum, signifance levels, and cone of influence
    # ! Period scale is logarithmic
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    power_spec = bx.contourf(
        t,
        np.log2(period),
        np.log2(power),
        np.log2(levels),
        extend="both",
        cmap="jet",
    )

    extent = [t.min(), t.max(), 0, max(period)]
    bx.contour(
        t, np.log2(period), sig95, [-99, 1], colors="k", linewidths=3, extent=extent
    )
    bx.fill(
        np.concatenate([t, t[-1:] + DT, t[-1:] + DT, t[:1] - DT, t[:1] - DT]),
        np.concatenate(
            [
                np.log2(coi),
                [levels[2]],
                np.log2(period[-1:]),
                np.log2(period[-1:]),
                [levels[2]],
            ]
        ).clip(
            min=-2.5
        ),  # ! To keep cone of influence from bleeding off graph
        "k",
        alpha=0.3,
        hatch="--",
    )

    # * Invert y axis
    bx.set_ylim(bx.get_ylim()[::-1])

    # * Set labels/title
    bx.set_xlabel("")
    bx.set_ylabel("Period (years)")
    #
    Yticks = 2 ** np.arange(
        np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))
    )
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)
    bx.set_title(LABEL)

    plt.show()


if __name__ == "__main__":
    main()

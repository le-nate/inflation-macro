"""Helper functions for wavelet transforms"""

import numpy as np
import numpy.typing as npt
import pycwt as wavelet


def standardize_data_for_xwt(
    s: npt.NDArray, detrend: bool = True, standardize=True, remove_mean: bool = False
) -> npt.NDArray:
    """
    Helper function for pre-processing data, specifically for wavelet analysis
    From: https://github.com/regeirk/pycwt/issues/35#issuecomment-809588607
    """

    # Derive the variance prior to any detrending
    std = s.std()
    smean = s.mean()

    if detrend and remove_mean:
        raise ValueError(
            "Only standardize by either removing secular trend or mean, not both."
        )

    # Remove the trend if requested
    if detrend:
        arbitrary_x = np.arange(0, s.size)
        p = np.polyfit(arbitrary_x, s, 1)
        snorm = s - np.polyval(p, arbitrary_x)
    else:
        snorm = s

    if remove_mean:
        snorm = snorm - smean

    # Standardize by the standard deviation
    if standardize:
        snorm = snorm / std

    return snorm


def normalize_xwt_results(
    signal_size: npt.NDArray,
    xwt_coeffs: npt.NDArray,
    coi: npt.NDArray,
    freqs: npt.NDArray,
    signif: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Normalize results for plotting"""
    period = 1 / freqs
    power = (np.abs(xwt_coeffs)) ** 2  ## Normalize wavelet power spectrum
    sig95 = np.ones([1, signal_size]) * signif[:, None]
    sig95 = power / sig95  ## Want where power / sig95 > 1
    coi_plot = np.concatenate(
        [np.log2(coi), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]
    )
    return period, power, sig95, coi_plot

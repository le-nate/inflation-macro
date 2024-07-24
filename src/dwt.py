"""
Smoothing of signals via wavelet reconstruction
"""

from dataclasses import dataclass, field
import logging
import sys
from typing import Dict, Generator, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import numpy.typing as npt
import pywt

from src.utils.logging_helpers import define_other_module_log_level
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Constants
MOTHER = pywt.Wavelet("sym12")


@dataclass
class DataForDWT:
    """Holds data for discrete wavelet transform"""

    y_values: npt.NDArray
    mother_wavelet: Type
    levels: float = None


@dataclass
class ResultsFromDWT:
    """Holds data for discrete wavelet transform
    `coeffs`: transform coefficients
    `levels`: transform levels applied
    `smoothed_signal_dict`: dictionary of coefficients for each time scale"""

    coeffs: npt.NDArray
    levels: float
    smoothed_signal_dict: Dict[int, Dict[str, npt.NDArray]] = field(
        default_factory=dict
    )

    def smooth_signal(
        self, y_values: npt.NDArray, mother_wavelet: Type
    ) -> Dict[int, Dict[str, npt.NDArray]]:
        """Generate smoothed signals based off wavelet coefficients for each pre-defined level"""
        ## Initialize dict for reconstructed signals
        signals_dict = {}

        ## Loop through levels and remove detail level component(s)
        # ! Note: signal_dict[l] provides the signal with levels <= l removed
        logger.debug(self.levels)
        for l in range(self.levels, 0, -1):
            print(f"s_{l} stored with key {l}")
            smooth_coeffs = self.coeffs.copy()
            signals_dict[l] = {}
            ## Set remaining detail coefficients to zero
            for coeff in range(1, l + 1):
                smooth_coeffs[-1 * coeff] = np.zeros_like(smooth_coeffs[-1 * coeff])
            signals_dict[l]["coeffs"] = smooth_coeffs
            # Reconstruct the signal using only the approximation coefficients
            reconst = pywt.waverec(smooth_coeffs, mother_wavelet)
            signals_dict[l]["signal"] = trim_signal(y_values, reconst)
        self.smoothed_signal_dict = signals_dict


def trim_signal(
    original_signal: npt.NDArray, reconstructed: npt.NDArray
) -> npt.NDArray:
    """Removes first or last observation for odd-numbered datasets"""
    ## Time series with uneven result in mismatched lengths with the reconstructed
    ## signal, so we remove a value from the approximated signal
    if len(original_signal) % 2 != 0:
        logger.warning("Trimming signal at beginning")
        return reconstructed[1:]
    return reconstructed


def run_dwt(dwt_data: Type[DataForDWT]) -> Type[ResultsFromDWT]:
    """Generate levels and coefficients from discrete wavelet transform with
    given wavelet function"""
    ## Define the wavelet type
    # w = dwt_data.mother_wavelet
    ## Choose the maximum decomposition level
    if dwt_data.levels is None:
        dwt_levels = pywt.dwt_max_level(
            data_len=len(dwt_data.y_values), filter_len=dwt_data.mother_wavelet.dec_len
        )
        print(
            f"""Max decomposition level of {dwt_levels} for time series length 
            of {len(dwt_data.y_values)}"""
        )
    else:
        dwt_levels = dwt_data.levels
    dwt_coeffs = pywt.wavedec(
        dwt_data.y_values, dwt_data.mother_wavelet, level=dwt_data.levels
    )
    return ResultsFromDWT(dwt_coeffs, dwt_levels)


def reconstruct_signal_component(
    signal_coeffs: list, wavelet: str, level: int
) -> tuple[dict, int]:
    """Reconstruct individual component"""
    component_coeffs = signal_coeffs.copy()
    for l in range(len(signal_coeffs)):
        if l == level:
            component_coeffs[l] = component_coeffs[l]
        else:
            component_coeffs[l] = np.zeros_like(component_coeffs[l])
    return pywt.waverec(component_coeffs, wavelet)


def plot_smoothing(
    smooth_signals: dict,
    original_t: npt.NDArray,
    original_y: npt.NDArray,
    ascending: bool = False,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, str]:
    """Graph series of smoothed signals with original signal"""

    # * Input name of time series
    name = input("Enter name of time series (to be included in plot)")
    while isinstance(name, str) is False:
        input("Please enter a name as text")

    fig = plt.figure(figsize=kwargs["figsize"])
    # * Loop through levels and add detail level components
    if ascending:
        order = reversed(list(smooth_signals.items()))
    else:
        order = list(smooth_signals.items())
    for i, (level, signal) in enumerate(order, 1):
        smooth_level = len(smooth_signals) - level
        ## Subplot for each smooth signal
        plt.subplot(len(smooth_signals), 1, i)
        plt.plot(original_t, original_y, label="Actual")
        plt.plot(original_t, signal["signal"])
        plt.xlabel("Year")
        plt.grid()
        plt.title(rf"Approximation: $S_{{j-{smooth_level}}}$")
        if i == 1:
            plt.legend()
    return fig, name


def main() -> None:
    """Run script"""

    raw_data = retrieve_data.get_insee_data("000857180")
    _, t, y = retrieve_data.clean_insee_data(raw_data)

    # * Create instance of DataForDWT class
    data_for_dwt = DataForDWT(y, MOTHER)

    # * Apply DWT and smooth signal
    results_from_dwt = run_dwt(data_for_dwt)
    results_from_dwt.smooth_signal(
        y_values=data_for_dwt.y_values, mother_wavelet=data_for_dwt.mother_wavelet
    )

    fig, fig_title = plot_smoothing(
        results_from_dwt.smoothed_signal_dict, t, y, figsize=(10, 10)
    )

    plt.xlabel("Year")
    plt.ylabel(f"{fig_title.capitalize()}")
    fig.suptitle(f"Wavelet smoothing of {fig_title.lower()}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

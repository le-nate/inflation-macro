"""Test data smoothing functions"""

# %%
import logging

import numpy as np
import pywt

from scripts import dwt
from analysis.helpers import define_other_module_log_level
from analysis import retrieve_data as rd
from scripts import simulation_consumption as sim

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)

# %%
print("Testing trim_signal catches odd-numbered signals")
test_signal = [x for x in range(1000)]
print(f"Signal length: {len(test_signal)}")
trim = dwt.trim_signal(test_signal, test_signal)
assert len(trim) == len(test_signal)
print(f"Signal length: {len(test_signal)}")
test_signal = [x for x in range(1001)]
print(f"Signal length: {len(test_signal)}")
trim = dwt.trim_signal(test_signal, test_signal)
assert len(trim) != len(test_signal)


# %%
print("Testing smooth signal with INSEE data")
raw_data = rd.get_insee_data("000857179")
_, t, y = rd.clean_insee_data(raw_data)

## Define the wavelet type
WAVELET = "db4"
smooth_signals, dwt_levels = dwt.smooth_signal(y, WAVELET)
# print(len(y), len(smooth_signals[dwt_levels]["signal"]))
assert len(smooth_signals[dwt_levels]["signal"]) == len(y)


# %%
print("Testing smooth signal with Banque de France data")
raw_data = rd.get_bdf_data(series_key="ICP.M.FR.N.000000.4.ANR")
_, t, y = rd.clean_bdf_data(raw_data)

## Define the wavelet type
WAVELET = "db4"
w = pywt.Wavelet(WAVELET)
smooth_signals, dwt_levels = dwt.smooth_signal(y, WAVELET)
print(len(y), len(smooth_signals[dwt_levels]["signal"]))
assert len(smooth_signals[dwt_levels]["signal"]) == len(y)


# %%
print("Testing smooth_signal with odd-numbered observations")
t = np.linspace(1, 1001, 1001)
y = sim.consumption(t)
# Perform wavelet decomposition
WAVELET = "sym12"  ## Wavelet type, Symmlet 12
smooth_signals, dwt_levels = dwt.smooth_signal(y, WAVELET)
print(len(y), len(smooth_signals[dwt_levels]["signal"]))
assert len(smooth_signals[dwt_levels]["signal"]) == len(y)

print("Wavelet smoothing testing complete.")

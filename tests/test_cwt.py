"""Test CWT functions"""

from __future__ import division
import logging
import sys

from datetime import datetime
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import pycwt as wavelet
from pycwt.helpers import find

from src.helpers import define_other_module_log_level
from src import retrieve_data as rd
from src.cwt import set_time_range, run_cwt

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MEASURE = "MICH"
DT = 1 / 12
MOTHER = wavelet.Morlet(f0=6)

logger.info("Test set_time_range")
raw_data = rd.get_fed_data(MEASURE, units="pc1", freqs="m")
_, t_date, dat = rd.clean_fed_data(raw_data)
t_test = set_time_range(t_date, DT)
assert len(t_test) == len(t_date)

logger.info("Test run_cwt")
power, period, _, _ = run_cwt(t_test, dat, MOTHER)
assert len(power) == len(period)

logger.info("CWT test complete!")

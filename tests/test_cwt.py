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
from src import retrieve_data, cwt

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MEASURE = "MICH"
raw_data = retrieve_data.get_fed_data(MEASURE, units="pc1", freqs="m")
_, t_date, dat = retrieve_data.clean_fed_data(raw_data)
data_for_cwt = cwt.DataForCWT(
    t_date, dat, cwt.MOTHER, cwt.DT, cwt.DJ, cwt.S0, cwt.LEVELS
)

logger.info("Test set_time_range")
t_test = data_for_cwt.time_range
assert len(t_test) == len(t_date)

logger.info("Test run_cwt")
results_from_cwt = cwt.run_cwt(data_for_cwt)
assert len(results_from_cwt.power) == len(results_from_cwt.period)

logger.info("CWT test complete!")

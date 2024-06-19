"""Test data retrieval functions"""

# %%
import logging

import numpy as np

from analysis.helpers import define_other_module_log_level
from analysis import retrieve_data as rd

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)

# %%
logging.info(
    "Testing get_fed_data, cleaned data, with 1-year expected inflation (EXPINF1YR)"
)
data = rd.get_fed_data("EXPINF1YR", freq="m")
assert isinstance(data, list)

# %%
logging.info(
    "Testing get_fed_data, no headers, with 1-year expected inflation (EXPINF1YR)"
)
data = rd.get_fed_data("EXPINF1YR", no_headers=False)
assert isinstance(data, dict)
# %%
logging.info(
    "Testing get_fed_data, cleaned data, with 1-year expected inflation (EXPINF1YR)"
)
data = rd.get_fed_data("EXPINF1YR", freq="m")
clean_t, clean_y = rd.clean_fed_data(data)
assert isinstance(clean_t[1], np.datetime64)

# %%
logging.info("Testing get_insee_data")
data = rd.get_insee_data("000857180")
assert isinstance(data, list)

logging.info("Testing clean_insee_data")
clean_t, clean_y = rd.clean_insee_data(data)
assert isinstance(clean_y, np.ndarray)
assert isinstance(clean_t, np.ndarray)


# %%
logging.info("Testing get_bdf_data, with series_key and dataset")
data = rd.get_bdf_data("ICP.M.FR.N.000000.4.ANR")
assert isinstance(data, list)

logging.info("Testing clean_bdf_data")
clean_t, clean_y = rd.clean_bdf_data(data)
assert isinstance(clean_y, np.ndarray)
assert isinstance(clean_t, np.ndarray)

logging.info("Data retrieval functions testing complete.")

# %%
logging.info("Testing get_world_bank_data")
data = rd.get_world_bank_data("FP.CPI.TOTL.ZG", "FR")
logging.info(data)

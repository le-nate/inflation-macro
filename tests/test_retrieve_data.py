"""Test data retrieval functions"""

# %%
import numpy as np
from analysis import retrieve_data as rd

# %%
print("Testing get_fed_data, cleaned data, with 1-year expected inflation (EXPINF1YR)")
data = rd.get_fed_data("EXPINF1YR", freq="m")
assert isinstance(data, list)

# %%
print("Testing get_fed_data, no headers, with 1-year expected inflation (EXPINF1YR)")
data = rd.get_fed_data("EXPINF1YR", no_headers=False)
assert isinstance(data, dict)
# %%
print("Testing get_fed_data, cleaned data, with 1-year expected inflation (EXPINF1YR)")
data = rd.get_fed_data("EXPINF1YR", freq="m")
clean_t, clean_y = rd.clean_fed_data(data)
assert isinstance(clean_t[1], np.datetime64)

# %%
print("Testing get_insee_data")
data = rd.get_insee_data("000857180")
assert isinstance(data, list)

print("Testing clean_insee_data")
clean_t, clean_y = rd.clean_insee_data(data)
assert isinstance(clean_y, np.ndarray)
assert isinstance(clean_t, np.ndarray)


# %%
print("Testing get_bdf_data, with series_key and dataset")
data = rd.get_bdf_data("ICP.M.FR.N.000000.4.ANR")
assert isinstance(data, list)

print("Testing clean_bdf_data")
clean_t, clean_y = rd.clean_bdf_data(data)
assert isinstance(clean_y, np.ndarray)
assert isinstance(clean_t, np.ndarray)

print("Data retrieval functions testing complete.")

# %%
print("Testing get_world_bank_data")
data = rd.get_world_bank_data("FP.CPI.TOTL.ZG", "FR")
print(data)

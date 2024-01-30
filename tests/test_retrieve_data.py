"""Test data retrieval functions"""
# from analysis.api.creds import *
from analysis import retrieve_data

print("Testing get_fed_data, cleaned data, with 1-year expected inflation (EXPINF1YR)")
data = retrieve_data.get_fed_data("EXPINF1YR", freq="m")
assert isinstance(data, list)

print(
    "Testing get_fed_data, no cleaned data, with 1-year expected inflation (EXPINF1YR)"
)
data = retrieve_data.get_fed_data("EXPINF1YR", clean_data=False)
assert isinstance(data, dict)

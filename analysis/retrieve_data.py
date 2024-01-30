"""Retrieves data for analysis via API from statistics agencies and central banks"""

import requests

from inflation.api import creds


def get_fed_data(series, api_key, **kwargs):
    """Retrieve data series from FRED database and convert to time series if desired"""
    # import requests
    # import pandas as pd

    ## API GET request
    apiKey = f"&api_key={api_key}"
    seriesId = f"series_id={series}"
    units = kwargs.get("units", None)
    freq = kwargs.get("freq", None)
    errors = kwargs.get("errors", None)
    url = f"https://api.stlouisfed.org/fred/series/observations?{seriesId}&{apiKey}"

    final_url = url + f"&units={units}" if units is not None else url
    final_url = final_url + f"&frequency={freq}" if freq is not None else final_url
    final_url = final_url + "&file_type=json"
    # if units is not None:
    #     url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}&units={units}&file_type=json"
    # else:
    #     url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={api_key}&file_type=json"
    # if freq is not None:

    print(f"Requesting {series}")
    r = requests.get(final_url, timeout=5)
    resource = r.json()

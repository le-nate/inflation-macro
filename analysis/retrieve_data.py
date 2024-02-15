"""Retrieve data for analysis via API from statistics agencies and central banks"""

import requests

from .api.creds import FED_KEY


def get_fed_data(series, clean_data=True, **kwargs):
    """Retrieve data series from FRED database and convert to time series if desired
    :param str: series Fed indicator's code (e.g. EXPINF1YR, for 1-year expected inflation)
    :param bool: clean_data Remove headers in json

    Some series codes:
    - Michigan Perceived Inflation (MICH)
    - 1-Year Expected Inflation (EXPINF1YR)
    - US CPI (CPIAUCSL) -- use with `units="pc1", freq="m"`
    - Personal Savings Rate (PSAVERT)
    - Personal Consumption Expenditure (PCE) -- use with `units='pc1'`
    """

    ## API GET request
    base_url = "https://api.stlouisfed.org/fred/series/observations"

    units = kwargs.get("units", None)
    freq = kwargs.get("freq", None)

    ## Request parameters
    params = {
        "api_key": FED_KEY,
        "series_id": series,
        "units": units,
        "freq": freq,
        "file_type": "json",
    }

    ## Remove parameters with None
    params = {k: v for k, v in params.items() if v is not None}

    ## Create final url for request
    final_url = f"{base_url}?{'&'.join([f'{k}={v}' for k, v in params.items()])}"

    ## Make request
    try:
        print(f"Requesting {series}")
        r = requests.get(final_url, timeout=5)
        r.raise_for_status()  # Raise an exception for 4XX and 5XX HTTP status codes
        resource = r.json()["observations"] if clean_data is True else r.json()
        print(f"Retrieved {series}")
        return resource
    except requests.Timeout:
        print("Timeout error: The request took too long to complete.")
        return None
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

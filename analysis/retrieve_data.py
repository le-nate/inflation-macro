"""Retrieve data for analysis via API from statistics agencies and central banks"""

import dateparser
from dotenv import load_dotenv
import json
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import requests
import wbgapi as wb
import xmltodict

## Retrieve API credentials
load_dotenv()
BDF_KEY = os.getenv("BDF_KEY")
FED_KEY = os.getenv("FED_KEY")
INSEE_AUTH = os.getenv("INSEE_AUTH")


def get_fed_data(series: str, no_headers: bool = True, **kwargs) -> str:
    """Retrieve data series from FRED database and convert to time series if desired
    :param str: series Fed indicator's code (e.g. EXPINF1YR, for 1-year expected inflation)
    :param bool: no_headers Remove headers in json

    Some series codes:
    - Michigan Perceived Inflation (`"MICH"`)
    - 1-Year Expected Inflation (`"EXPINF1YR"`)
    - US CPI (`"CPIAUCSL", units="pc1", freq="m"`)
    - Personal Savings Rate (`"PSAVERT"`)
    - Personal Consumption Expenditure (`"PCE", units="pc1"`)
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
        resource = r.json()["observations"] if no_headers is True else r.json()
        print(f"Retrieved {series}")
        return resource
    except requests.Timeout:
        print("Timeout error: The request took too long to complete.")
        return None
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None


def clean_fed_data(json_data: str) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    """Convert Fed data to time and endogenous variables (t, y)"""

    ## Convert to dataframe
    df = pd.DataFrame(json_data)
    print(df.info(verbose=True), "\n")

    ## Convert dtypes
    df.replace(".", np.nan, inplace=True)
    df.dropna(inplace=True)
    df["value"] = pd.to_numeric(df["value"])
    df["date"] = pd.to_datetime(df["date"])

    ## Drop extra columns
    df = df[["date", "value"]]

    t = df["date"].to_numpy()
    y = df["value"].to_numpy()

    return df, t, y


def get_insee_data(series_id: str) -> list:
    """
    Retrieve data (Series_BDM) from INSEE API

    Some series codes:
    - 'Expected inflation' `"000857180"`,
    - 'Perceived inflation' `"000857179"`,
    - 'Expected savings' `"000857186"`,
    - 'Expected consumption' `"000857181"`,
    """
    url = f"https://api.insee.fr/series/BDM/V1/data/SERIES_BDM/{series_id}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {INSEE_AUTH}",
    }
    response = requests.get(url, headers=headers, timeout=10)
    decoded_response = response.content.decode("utf-8")
    response_json = json.loads(json.dumps(xmltodict.parse(decoded_response)))
    series_title = response_json["message:StructureSpecificData"]["message:DataSet"][
        "Series"
    ]["@TITLE_FR"]
    response_data = response_json["message:StructureSpecificData"]["message:DataSet"][
        "Series"
    ]["Obs"]
    print(f"Retrieved {series_title}. \n{len(response_data)} observations\n")

    return response_data


def clean_insee_data(
    data: list, ascending: bool = True
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    """Convert INSEE data to time and endogenous variables (t, y)"""
    df = pd.DataFrame(data)
    # Convert data types
    df["@TIME_PERIOD"] = pd.to_datetime(df["@TIME_PERIOD"])
    df["@OBS_VALUE"] = df["@OBS_VALUE"].astype(float)
    if ascending is True and df["@TIME_PERIOD"].iloc[-1] < df["@TIME_PERIOD"].iloc[0]:
        df = df[::-1]
    elif (
        ascending is False and df["@TIME_PERIOD"].iloc[-1] > df["@TIME_PERIOD"].iloc[0]
    ):
        df = df[::-1]
    else:
        pass
    t = df["@TIME_PERIOD"].to_numpy()
    y = df["@OBS_VALUE"].to_numpy()

    return df, t, y


def get_bdf_data(series_key: str, dataset: str = "ICP", **kwargs) -> str:
    """Retrieve data from Banque de France API
    Measured inflation: `'ICP.M.FR.N.000000.4.ANR'`
    """

    ## API GET request
    data_type = kwargs.get("data_type", "data")
    req_format = kwargs.get("format", "json")
    headers = {"accept": "application/json"}

    base_url = "https://api.webstat.banque-france.fr/webstat-fr/v1/"

    params = {
        "data_type": data_type,
        "dataset": dataset,
        "series_key": series_key,
    }

    ## Remove parameters with None
    params = {k: v for k, v in params.items() if v is not None}

    ## Create final url for request
    final_url = f"{base_url}{'/'.join([f'{v}' for k, v in params.items()])}?client_id={BDF_KEY}&format={req_format}"

    print(f"Requesting {series_key}")
    r = requests.get(final_url, headers=headers, timeout=5)
    print(r)
    response = r.json()
    response = response["seriesObs"][0]["ObservationsSerie"]["observations"]

    return response


def clean_bdf_data(
    data: list, ascending: bool = True
) -> tuple[pd.DataFrame, npt.NDArray, npt.NDArray]:
    """Convert list of dicts data from Banque de France to lists for t and y"""
    ## Dictionary of observations
    dict_obs = {
        "periodId": [],
        "periodFirstDate": [],
        "periodName": [],
        "value": [],
    }
    for i in data:
        obs = i["ObservationPeriod"]
        for k, v in dict_obs.items():
            v.append(obs[k])

    ## Convert to df
    df = pd.DataFrame(dict_obs)
    if (
        ascending is True
        and df["periodFirstDate"].iloc[-1] < df["periodFirstDate"].iloc[0]
    ):
        df = df[::-1]
    elif (
        ascending is False
        and df["periodFirstDate"].iloc[-1] > df["periodFirstDate"].iloc[0]
    ):
        df = df[::-1]
    else:
        pass
    df["periodFirstDate"] = pd.to_datetime(df["periodFirstDate"])
    t = df["periodFirstDate"].to_numpy()
    y = df["value"].to_numpy()

    return df, t, y


def get_world_bank_data(series_id: str, country: str) -> str:
    """Retrieve inflation data by country
    Inflation rate (annual %): 'FP.CPI.TOTL.ZG'
    CPI (2010 = 100): 'FP.CPI.TOTL'
    France: 'FR'
    United States: 'US'
    """
    base_url = f"https://api.worldbank.org/v2/indicator/{series_id}?locations={country}?format=json"
    response = requests.get(base_url, timeout=5)
    print(response)
    return response.json()


def data_to_time_series(df, index_column, measure=None):
    """Convert dataframe to time series"""
    if measure is not None:
        df = df[[index_column, "value"]][df["measure"] == measure]
    else:
        df = df[[index_column, "value"]]
    ## Set date as index
    df.set_index(index_column, inplace=True)
    df = df.astype(float)
    return df

"""Script to combine and standardize yearly CAMME responses from separate csv files"""

import logging
import os
from pathlib import Path
import time
from typing import Dict, Generator, List, Union

import dateparser
import numpy as np
import numpy.typing as npt
import pandas as pd

from helpers import nested_dict_values, nested_list_values
from constants.camme import (
    IGNORE_HOUSING,
    IGNORE_HOUSING_YEARS,
    IGNORE_SUPPLEMENTS,
    VARS_DICT,
)
import data.camme


# * Define logging threshold
logging.basicConfig(level=logging.DEBUG)

# * Get data directory folder
parent_dir = Path(__file__).parents[1]
data_dir = parent_dir / "data" / "camme"


def retrieve_folders(path: Union[str, os.PathLike]) -> Dict[
    str,
    Dict[Dict[str, Union[str, os.PathLike]], Dict[str, List[Union[str, os.PathLike]]]],
]:
    """Loop through the csv subdirectories"""
    dir_dict = {}
    for sub_dir in path.rglob("Csv"):
        dir_dict[sub_dir.parents[0].name] = {"path": sub_dir, "csv": []}
    return dir_dict


def retrieve_csv_files(
    dir_dict: Dict[str, Union[str, os.PathLike]]
) -> Dict[str, Union[str, os.PathLike]]:
    """Add only standard questionnaire csv file paths to dictionary"""
    for year in dir_dict:
        total_files = sum(len(files) for _, _, files in os.walk(dir_dict[year]["path"]))
        logging.info("There are %s total files for year %s", total_files, year)
        for file in dir_dict[year]["path"].rglob("*"):
            if any(supp in file.name for supp in IGNORE_SUPPLEMENTS):
                pass
            elif (
                any(y in file.parents[1].name for y in IGNORE_HOUSING_YEARS)
                and IGNORE_HOUSING in file.name
            ):
                pass
            else:
                dir_dict[year]["csv"].append(file)
        files_kept_count = len(dir_dict[year]["csv"])
        logging.info(
            "%s of %s files maintained for year %s", files_kept_count, total_files, year
        )
    return dir_dict


def convert_to_year_dataframe(
    dir_dict: Dict[str, Union[str, os.PathLike]]
) -> Dict[str, pd.DataFrame]:
    """Combine all dataframes into one for all years"""
    df_dict = {}
    for year in dir_dict:
        logging.debug(year)
        df_complete = []
        for table in dir_dict[year]["csv"]:
            df = pd.read_csv(table, delimiter=";", encoding="latin-1")
            df_complete.append(df)
        df_dict[year] = pd.concat(df_complete)
        df_dict[year]["year"] = int(year)
        logging.debug("DataFrame shape: %s", df_dict[year].shape)
    return df_dict


def main() -> None:
    """Run script"""
    logging.info("Retrieving folders")
    camme_csv_folders = retrieve_folders(data_dir)
    logging.info("Retrieving CSV files")
    camme_csv_folders = retrieve_csv_files(camme_csv_folders)
    logging.info("Converting to DataFrames")
    start = time.time()
    dfs = convert_to_year_dataframe(camme_csv_folders)
    end = time.time()
    dtime = end - start
    logging.debug("Elapsed time to convert to DataFrames: %s", dtime)
    ## Elapsed time before change: 11.67s
    cols = list(nested_dict_values(VARS_DICT))
    cols = list(nested_list_values(cols))
    cols.insert(0, "year")
    logging.debug(cols)
    print(dfs["2014"][cols].head())

    # TODO add column for year


if __name__ == "__main__":
    main()

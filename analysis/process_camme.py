"""Script to combine and standardize yearly CAMME responses from separate csv files"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Union

import dateparser
import numpy as np
import numpy.typing as npt
import pandas as pd

import data.camme

# * Define logging threshold
logging.basicConfig(level=logging.DEBUG)

# * Get data directory folder
parent_dir = Path(__file__).parents[1]
data_dir = parent_dir / "data" / "camme"

# * Survey waves to ignore
IGNORE_SUPPLEMENTS = ["be", "cnle", "cov", "pf"]
IGNORE_HOUSING = "log"
# Only these years had separate housing surveys
IGNORE_HOUSING_YEARS = ["2016", "2017"]

# * Variables and corresponding column names (change over course of series)
# * Variables used in Andrade et al. (2023)
VARS_DICT = {
    "inf_exp_qual": {},
    "inf_exp_val": {},
    "consump_past": {},
    "consump_general": {
        "2014": "ACHATS",
    },
}


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
        logging.info(f"There are {total_files} total files for year {year}")
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
            f"{files_kept_count} of {total_files} files maintained for year {year}"
        )
    return dir_dict


def convert_to_dataframe(
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
        logging.debug(df_dict[year].shape)
    return df_dict


def main() -> None:
    """Run script"""
    camme_csv_folders = retrieve_folders(data_dir)
    camme_csv_folders = retrieve_csv_files(camme_csv_folders)
    dfs = convert_to_dataframe(camme_csv_folders)


if __name__ == "__main__":
    main()

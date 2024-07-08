"""Statistical tests for time series analysis"""

import logging
from pathlib import Path
import sys

from typing import Dict, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm

from src.helpers import define_other_module_log_level
from src import ids
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# TODO Calculate mean & std

# TODO Calculate skewness

# TODO Calculate kurtosis

# TODO Calculate Jarque-Berra

# TODO Calculate Ljung-Box

# TODO Generate correlation matrix


# TODO Generate summary table
def create_summary_table(
    data_dict: Dict[str, Dict[str, float]], export_table: bool = True
) -> Union[Tuple[pd.DataFrame, None], pd.DataFrame]:
    """Create table with descriptive statistics for all datasets with option to export"""
    df = pd.DataFrame(data_dict)
    if export_table:
        # * Get data directory folder
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "descriptive_stats.html"

        return df, df.to_html(export_file)
    return df


def main() -> None:
    """Run script"""
    # * Retrieve data

    ## Inflation
    raw_data = retrieve_data.get_fed_data(ids.US_CPI, units="pc1")
    inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf.rename(columns={"value": "inflation"}, inplace=True)

    ## Inflation expectations
    raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": "expectation"}, inplace=True)

    ## Non-durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_NONDURABLES_CONSUMPTION, units="pc1")
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

    ## Durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_DURABLES_CONSUMPTION, units="pc1")
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": "durable"}, inplace=True)

    ## Personal savings rate
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS)
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": "savings"}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    us_data = inf.merge(inf_exp, how="left")
    us_data = us_data.merge(nondur_consump, how="left")
    us_data = us_data.merge(dur_consump, how="left")
    us_data = us_data.merge(save, how="left")

    results_dict = us_data.describe(percentiles=[0.5]).to_dict()
    results_df = create_summary_table(results_dict)
    print(results_df)


if __name__ == "__main__":
    main()

"""Statistical tests for time series analysis"""

import logging
from pathlib import Path
import sys

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.graphics.tsaplots
import statsmodels.stats.diagnostic

from constants import ids
from src.helpers import add_real_value_columns
from src.logging_helpers import define_other_module_log_level
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("info")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Define constant currency years
CONSTANT_DOLLAR_DATE = "2017-12-01"

DESCRIPTIVE_STATS = [
    "count",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "Jarque-Bera",
    "Shapiro-Wilk",
    "Ljung-Box",
]
NORMALITY_TESTS = {"Jarque-Bera": stats.jarque_bera, "Shapiro-Wilk": stats.shapiro}
HYPOTHESIS_THRESHOLD = [0.1, 0.05, 0.001]


def parse_results_dict(
    complete_dict: Dict[str, Dict[str, Union[float, str]]], stats_to_keep: List[str]
) -> Dict[str, Dict[str, float]]:
    """Extract only required statistics from dictionary"""
    parsed_dict = {}
    for measure, result_dict in complete_dict.items():
        parsed_dict[measure] = {stat: result_dict[stat] for stat in stats_to_keep}
    return parsed_dict


def include_statistic(
    statistic: str,
    statistic_data: Dict[str, float],
    results_dict: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Add skewness data for each measure"""
    for measure, result_dict in results_dict.items():
        result_dict[statistic] = statistic_data[measure]
    return results_dict


def add_p_value_stars(
    test_statistic: Union[int, float], p_value: float, hypothesis_threshold: List[float]
) -> str:
    """Add stars (*) for each p value threshold that the test statistic falls below"""
    star_test_statistic = str(test_statistic)
    for p_threshold in sorted(hypothesis_threshold):
        star_test_statistic += "*" if p_value <= p_threshold else ""
    return star_test_statistic


def test_normality(
    normality_test: str,
    data: pd.DataFrame,
    date_column: str = "date",
    add_pvalue_stars: bool = False,
) -> Dict[str, str]:
    """Generate dictionary with Jarque-Bera test results for each dataset"""
    results_dict = {}
    cols_to_test = data.drop(date_column, axis=1).columns.to_list()
    for col in cols_to_test:
        x = data[col].dropna().to_numpy()
        test_stat, p_value = NORMALITY_TESTS[normality_test](x)
        if add_pvalue_stars:
            result = add_p_value_stars(test_stat, p_value, HYPOTHESIS_THRESHOLD)
        results_dict[col] = result
    return results_dict


def conduct_ljung_box(
    data: pd.DataFrame,
    lags: List[int],
    date_column: str = "date",
    add_pvalue_stars: bool = False,
) -> Dict[str, str]:
    """Generate dictionary with Ljung-Box test results for each dataset"""
    results_dict = {}
    cols_to_test = data.drop(date_column, axis=1).columns.to_list()
    for col in cols_to_test:
        test_results = statsmodels.stats.diagnostic.acorr_ljungbox(data[col], lags=lags)
        test_stat, p_value = (
            test_results["lb_stat"].iat[0],
            test_results["lb_pvalue"].iat[0],
        )
        if add_pvalue_stars:
            result = add_p_value_stars(test_stat, p_value, HYPOTHESIS_THRESHOLD)
        results_dict[col] = result
    return results_dict


def correlation_matrix_pvalues(
    data: pd.DataFrame,
    hypothesis_threshold: List[float],
    decimals: int = 2,
    display: bool = False,
    export_table: bool = False,
) -> pd.DataFrame:
    """Calculate pearson correlation and p-values and add asterisks
    to relevant values in table"""
    rho = data.corr()
    pval = data.corr(method=lambda x, y: stats.pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(
        lambda x: "".join(["*" for threshold in hypothesis_threshold if x <= threshold])
    )
    corr_matrix = rho.round(decimals).astype(str) + p
    if display:
        cols = data.columns.to_list()
        print(f"P-values benchmarks: {hypothesis_threshold}")
        for c in cols:
            print(c)
            print(f"{c} p-values: \n{pval[c]}")
    if export_table:
        # * Get results directory
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "correlation_matrix.html"
        corr_matrix.to_html(export_file)
    return corr_matrix


def create_summary_table(
    data_dict: Dict[str, Dict[str, Union[float, str]]], export_table: bool = False
) -> pd.DataFrame:
    """Create table with descriptive statistics for all datasets with option to export"""
    df = pd.DataFrame(data_dict)
    if export_table:
        # * Get results directory
        parent_dir = Path(__file__).parents[1]
        export_file = parent_dir / "results" / "descriptive_stats.html"

        df.to_html(export_file)
    return df


def main() -> None:
    """Run script"""
    # * Retrieve data

    ## CPI
    raw_data = retrieve_data.get_fed_data(ids.US_CPI)
    cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
    cpi.rename(columns={"value": "cpi"}, inplace=True)

    ## Inflation
    raw_data = retrieve_data.get_fed_data(ids.US_CPI, units="pc1")
    inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf.rename(columns={"value": "inflation"}, inplace=True)

    ## Inflation expectations
    raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": "expectation"}, inplace=True)

    ## Non-durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_NONDURABLES_CONSUMPTION)
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

    ## Durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_DURABLES_CONSUMPTION)
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": "durable"}, inplace=True)

    ## Personal savings rate
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS)
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": "savings"}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    us_data = cpi.merge(inf, how="left")
    us_data = us_data.merge(inf_exp, how="left")
    us_data = us_data.merge(nondur_consump, how="left")
    us_data = us_data.merge(dur_consump, how="left")
    us_data = us_data.merge(save, how="left")

    # * Drop NaNs
    us_data.dropna(inplace=True)

    # * Add real value columns
    logger.info(
        "Using constant dollars from %s, CPI: %s",
        CONSTANT_DOLLAR_DATE,
        us_data[us_data["date"] == pd.Timestamp(CONSTANT_DOLLAR_DATE)]["cpi"].iat[0],
    )
    us_data = add_real_value_columns(
        data=us_data,
        nominal_columns=["nondurable", "durable", "savings"],
        cpi_column="cpi",
        constant_date=CONSTANT_DOLLAR_DATE,
    )

    print(us_data.head())

    results = us_data.describe(percentiles=[0.5]).to_dict()
    skewness = us_data.skew(numeric_only=True).to_dict()
    kurtosis = us_data.kurtosis(numeric_only=True).to_dict()
    jarque_bera = test_normality(
        normality_test="Jarque-Bera",
        data=us_data,
        date_column="date",
        add_pvalue_stars=True,
    )
    shapiro_wilk = test_normality(
        normality_test="Shapiro-Wilk",
        data=us_data,
        date_column="date",
        add_pvalue_stars=True,
    )
    ljung_box = conduct_ljung_box(
        data=us_data, lags=[15], date_column="date", add_pvalue_stars=True
    )
    results = include_statistic(
        statistic="skewness", statistic_data=skewness, results_dict=results
    )
    results = include_statistic(
        statistic="kurtosis", statistic_data=kurtosis, results_dict=results
    )
    results = include_statistic(
        statistic="Jarque-Bera", statistic_data=jarque_bera, results_dict=results
    )
    results = include_statistic(
        statistic="Shapiro-Wilk", statistic_data=shapiro_wilk, results_dict=results
    )
    results = include_statistic(
        statistic="Ljung-Box", statistic_data=ljung_box, results_dict=results
    )
    results = parse_results_dict(results, DESCRIPTIVE_STATS)
    results_df = create_summary_table(results, export_table=True)
    print(results_df)

    us_corr = correlation_matrix_pvalues(
        data=us_data,
        hypothesis_threshold=HYPOTHESIS_THRESHOLD,
        decimals=2,
        display=False,
        export_table=True,
    )
    print(us_corr)

    us_data.plot.hist(bins=150, subplots=True, legend=True)
    _, axs = plt.subplots(5)
    for ax, c in zip(axs, us_data.drop("date", axis=1).columns.to_list()):
        statsmodels.graphics.tsaplots.plot_acf(us_data[c], lags=36, ax=ax)
        ax.title.set_text(c)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

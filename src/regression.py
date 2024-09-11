"""Conduct regression using denoised data via DWT"""

import logging
import sys

from typing import Dict, Type

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import numpy.typing as npt
import pandas as pd
import pywt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.iolib.summary2

from constants import ids, results_configs
from src import dwt
from src.utils import helpers
from src.utils.logging_helpers import define_other_module_log_level
from src.utils.wavelet_helpers import align_series
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# ! Define mother wavelet
MOTHER = pywt.Wavelet("db4")


def simple_regression(
    data: pd.DataFrame, x_var: str, y_var: str, add_constant: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Perform simple linear regression on data"""
    x_simp = data[x_var]
    y_simp = data[y_var]
    if add_constant:
        x_simp = sm.add_constant(x_simp)
    model = sm.OLS(y_simp, x_simp)
    results = model.fit()
    return results


def wavelet_approximation(
    smooth_t_dict: Dict[int, Dict[str, npt.NDArray]],
    original_y: npt.NDArray,
    levels: int,
    add_constant: bool = True,
    verbose: bool = False,
) -> Dict[int, Type[sm.regression.linear_model.RegressionResultsWrapper]]:
    """Regresses smooth components"""
    regressions_dict = {}
    crystals = list(range(1, levels + 1))
    for c in crystals:
        x_c = smooth_t_dict[c]["signal"]
        if add_constant:
            x_c = sm.add_constant(x_c)
        model = sm.OLS(original_y, x_c)
        results = model.fit()
        if verbose:
            print("\n\n")
            print(f"-----Smoothed model, Removing D_{list(range(1, c+1))}-----")
            print("\n")
            print(results.summary())
        regressions_dict[c] = results
    return regressions_dict


def time_scale_regression(
    input_coeffs: npt.NDArray,
    output_coeffs: npt.NDArray,
    levels: int,
    mother_wavelet: str,
    add_constant: bool = True,
) -> Type[statsmodels.iolib.summary2.Summary]:
    """Regresses output on  input for each component vector S_J, D_J, ..., D_1,
    where J=levels"""
    regressions_dict = {}
    for j in range(levels + 1):
        if j == 0:
            vector_name = f"S_{levels}"
        else:
            vector_name = f"D_{levels - j + 1}"
        print(f"Regressing on component vector {vector_name}")
        # * Reconstruct each component vector indiviually
        input_j = dwt.reconstruct_signal_component(input_coeffs, mother_wavelet, j)
        output_j = dwt.reconstruct_signal_component(output_coeffs, mother_wavelet, j)

        # * Run regression
        if add_constant:
            input_j = sm.add_constant(input_j)
        model = sm.OLS(output_j, input_j)
        regressions_dict[vector_name] = model.fit()
    results = statsmodels.iolib.summary2.summary_col(
        list(regressions_dict.values()),
        stars=True,
        model_names=list(regressions_dict),
    )
    return results


def plot_compare_components(
    a_label: str,
    b_label: str,
    a_coeffs: npt.NDArray,
    b_coeffs: npt.NDArray,
    time: npt.NDArray,
    levels: int,
    wavelet: str,
    # ascending: bool = False,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Plot each series component separately"""
    fig, ax = plt.subplots(levels + 1, 1, **kwargs)
    smooth_component_x = dwt.reconstruct_signal_component(a_coeffs, wavelet, 0)
    smooth_component_y = dwt.reconstruct_signal_component(b_coeffs, wavelet, 0)
    logger.warning(
        "lengths x: %s, y: %s, t: %s",
        len(smooth_component_x),
        len(smooth_component_y),
        len(time),
    )

    # * Align array legnths
    if len(smooth_component_x) != len(time):
        smooth_component_x = align_series(time, smooth_component_x)
    if len(smooth_component_y) != len(time):
        smooth_component_y = align_series(time, smooth_component_y)
    ax[0].plot(time, smooth_component_x, label=a_label)
    ax[0].plot(time, smooth_component_y, label=b_label)
    ax[0].set_title(rf"$S_{{{levels}}}$")

    components = {}
    for l in range(1, levels + 1):
        components[l] = {}
        for c, c_coeffs in zip([a_label, b_label], [a_coeffs, b_coeffs]):
            components[l][c] = dwt.reconstruct_signal_component(c_coeffs, wavelet, l)
            if len(time) != len(components[l][c]):
                components[l][c] = align_series(time, components[l][c])
            ax[l].plot(time, components[l][c], label=c)
            ax[l].set_title(rf"$D_{{{levels + 1 - l}}}$")
    plt.legend(loc="upper left", frameon=False)
    return fig


def main() -> None:
    """Run script"""
    # * CPI
    raw_data = retrieve_data.get_fed_data(ids.US_CPI)
    cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
    cpi.rename(columns={"value": ids.CPI}, inplace=True)

    # * Inflation rate
    raw_data = retrieve_data.get_fed_data(ids.US_CPI, units="pc1", freq="m")
    measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
    measured_inf.rename(columns={"value": ids.INFLATION}, inplace=True)

    # * Inflation expectations
    raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
    inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": ids.EXPECTATIONS}, inplace=True)

    # * Non-durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_NONDURABLES_CONSUMPTION)
    nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": ids.NONDURABLES}, inplace=True)

    # * Durables consumption, monthly
    raw_data = retrieve_data.get_fed_data(ids.US_DURABLES_CONSUMPTION)
    dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": ids.DURABLES}, inplace=True)

    # * Non-durables consumption change, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_NONDURABLES_CONSUMPTION, units="pc1", freq="m"
    )
    nondur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    nondur_consump_chg.rename(columns={"value": ids.NONDURABLES_CHG}, inplace=True)

    # * Durables consumption change, monthly
    raw_data = retrieve_data.get_fed_data(
        ids.US_DURABLES_CONSUMPTION, units="pc1", freq="m"
    )
    dur_consump_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    dur_consump_chg.rename(columns={"value": ids.DURABLES_CHG}, inplace=True)

    # * Personal savings
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS)
    save, _, _ = retrieve_data.clean_fed_data(raw_data)
    save.rename(columns={"value": ids.SAVINGS}, inplace=True)

    # * Personal savings change
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS, units="pc1", freq="m")
    save_chg, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_chg.rename(columns={"value": ids.SAVINGS_CHG}, inplace=True)

    # * Personal savings rate
    raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS_RATE)
    save_rate, _, _ = retrieve_data.clean_fed_data(raw_data)
    save_rate.rename(columns={"value": ids.SAVINGS_RATE}, inplace=True)

    # * Merge dataframes to align dates and remove extras
    dataframes = [
        cpi,
        measured_inf,
        inf_exp,
        nondur_consump,
        nondur_consump_chg,
        dur_consump,
        dur_consump_chg,
        save,
        save_chg,
        save_rate,
    ]
    us_data = helpers.combine_series(dataframes, on=[ids.DATE], how="left")

    # # * Remove rows without data for all measures
    us_data.dropna(inplace=True)

    # * Add real value columns
    logger.info(
        "Using constant dollars from %s, CPI: %s",
        results_configs.CONSTANT_DOLLAR_DATE,
        us_data[
            us_data[ids.DATE] == pd.Timestamp(results_configs.CONSTANT_DOLLAR_DATE)
        ][ids.CPI].iat[0],
    )
    us_data = helpers.add_real_value_columns(
        data=us_data,
        nominal_columns=[ids.NONDURABLES, ids.DURABLES, ids.SAVINGS],
        cpi_column=ids.CPI,
        constant_date=results_configs.CONSTANT_DOLLAR_DATE,
    )
    df = helpers.calculate_diff_in_log(
        data=us_data,
        columns=[
            ids.CPI,
            ids.EXPECTATIONS,
            ids.NONDURABLES,
            ids.DURABLES,
            ids.SAVINGS,
            ids.REAL_NONDURABLES,
            ids.REAL_DURABLES,
            ids.REAL_SAVINGS,
        ],
    )

    print(
        f"""Inflation expectations observations: {len(inf_exp)}, \n
        Non-durables consumption observations: {len(nondur_consump)}, \n
        Durables consumption observations: {len(dur_consump)}, \n
        Savings observation {len(save)}.\nNew dataframe lengths: {len(df)}"""
    )
    print(df.head(), "\n", df.tail())
    print("--------------------------Descriptive stats--------------------------\n")
    print(df.describe())

    # sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

    # * Wavelet decomposition
    t = df["date"].to_numpy()

    ## Create objects for DWT
    exp_dwt = dwt.DataForDWT(df[f"diff_log_{ids.EXPECTATIONS}"].to_numpy(), MOTHER)
    nondur_consump_dwt = dwt.DataForDWT(
        df[f"diff_log_{ids.NONDURABLES}"].to_numpy(), MOTHER
    )
    dur_consump_dwt = dwt.DataForDWT(df[f"diff_log_{ids.DURABLES}"].to_numpy(), MOTHER)
    save_dwt = dwt.DataForDWT(df[f"diff_log_{ids.SAVINGS}"].to_numpy(), MOTHER)

    logger.debug("exp mother wavelet %s", type(exp_dwt.mother_wavelet))
    results_exp_dwt = dwt.run_dwt(exp_dwt)
    results_nondur_consump_dwt = dwt.run_dwt(nondur_consump_dwt)
    results_dur_consump_dwt = dwt.run_dwt(dur_consump_dwt)
    results_save_dwt = dwt.run_dwt(save_dwt)

    df_melt = pd.melt(df, ["date"])
    df_melt.rename(columns={"value": "%"}, inplace=True)

    # # * Plot log-by-log change amount frequency
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # sns.kdeplot(data=df_melt, x="%", hue="variable", ax=ax)

    # * Plot each series component separately
    _ = plot_compare_components(
        f"diff_log_{ids.EXPECTATIONS}",
        f"diff_log_{ids.NONDURABLES}",
        results_exp_dwt.coeffs,
        results_nondur_consump_dwt.coeffs,
        t,
        results_exp_dwt.levels,
        MOTHER,
        figsize=(10, 15),
        sharex=True,
    )

    _ = plot_compare_components(
        f"diff_log_{ids.EXPECTATIONS}",
        f"diff_log_{ids.DURABLES}",
        results_exp_dwt.coeffs,
        results_dur_consump_dwt.coeffs,
        t,
        results_exp_dwt.levels,
        MOTHER,
        figsize=(10, 15),
        sharex=True,
    )

    _ = plot_compare_components(
        f"diff_log_{ids.EXPECTATIONS}",
        f"diff_log_{ids.SAVINGS}",
        results_exp_dwt.coeffs,
        results_save_dwt.coeffs,
        t,
        results_exp_dwt.levels,
        MOTHER,
        figsize=(10, 15),
        sharex=True,
    )

    # # * Plot initial series
    # _, (bx) = plt.subplots(1, 1)
    # bx = sns.lineplot(data=df_melt, x="date", y="%", hue="variable", ax=bx)

    plt.show()


if __name__ == "__main__":
    main()

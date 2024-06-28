"""Conduct regression using denoised data via DWT"""

import logging
import sys

from typing import Dict, Type

import matplotlib.pyplot as plt
import matplotlib.figure
import numpy.typing as npt
import pandas as pd
import pywt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.iolib.summary2

from src import dwt
from src.helpers import define_other_module_log_level
from src import retrieve_data as rd

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def simple_regression(
    data: pd.DataFrame, x_var: str, y_var: str, add_constant: bool = True
):
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
    smooth_a_coeffs: npt.NDArray,
    smooth_b_coeffs: npt.NDArray,
    time: npt.NDArray,
    levels: int,
    wavelet: str,
    # ascending: bool = False,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Plot each series component separately"""
    fig, ax = plt.subplots(levels + 1, 1, **kwargs)
    smooth_component_x = dwt.reconstruct_signal_component(smooth_a_coeffs, wavelet, 0)
    smooth_component_y = dwt.reconstruct_signal_component(smooth_b_coeffs, wavelet, 0)

    ax[0].plot(time, smooth_component_x, label=a_label)
    ax[0].plot(time, smooth_component_y, label=b_label)
    ax[0].set_title(rf"$S_{{{levels}}}$")

    components = {}
    for l in range(1, levels + 1):
        components[l] = {}
        for c, c_coeffs in zip([a_label, b_label], [smooth_a_coeffs, smooth_b_coeffs]):
            components[l][c] = dwt.reconstruct_signal_component(c_coeffs, wavelet, l)
            ax[l].plot(time, components[l][c], label=c)
            ax[l].set_title(rf"$D_{{{levels + 1 - l}}}$")
    plt.legend(loc="upper left")
    return fig


# ! Define mother wavelet
MOTHER = pywt.Wavelet("db4")


# # %% [markdown]
# # # US data for comparison to Coibion et al. (2021)
# print(
#     """Coibion et al. (2021) find that inflation expectations have a positive
# relationship with nondurable and services consumption and a negative relationship
# with durable consumption. A 1% increase in inflation expectations correlates with
# a 1.8% increase in nondurables and services consumption and 1.5% decrease in
# durables consumption."""
# )

# # %% [markdown]
# # ## Get data

# # %%
# # * Inflation expectations
# raw_data = rd.get_fed_data("MICH")
# inf_exp, _, _ = rd.clean_fed_data(raw_data)
# ## Rename value column
# inf_exp.rename(columns={"value": "expectation"}, inplace=True)
# print("Descriptive stats for inflation expectations")
# print(inf_exp.describe())

# # %%
# # * Non-durables consumption, monthly
# raw_data = rd.get_fed_data("PCEND", units="pc1")
# nondur_consump, _, _ = rd.clean_fed_data(raw_data)
# nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)
# print("Descriptive stats for personal non-durables consumption")
# print(nondur_consump.describe())

# # %%
# # * Durables consumption, monthly
# raw_data = rd.get_fed_data("PCEDG", units="pc1")
# dur_consump, _, _ = rd.clean_fed_data(raw_data)
# dur_consump.rename(columns={"value": "durable"}, inplace=True)
# print("Descriptive stats for personal durables consumption")
# print(dur_consump.describe())

# # %%
# # * Merge dataframes to remove extra dates
# df = inf_exp.merge(nondur_consump, how="left")
# df = df.merge(dur_consump, how="left")
# print(
#     f"""Inflation expectations observations: {len(inf_exp)}, \nNon-durables
#       consumption observations: {len(nondur_consump)}, \nDurables
#       consumption observations: {len(dur_consump)}.\nNew dataframe lengths: {len(df)}"""
# )
# print(df.head(), "\n", df.tail())

# # %%
# sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

# # %% [markdown]
# # ## Wavelet decomposition

# # %%
# t = df["date"].to_numpy()
# x = df["expectation"].to_numpy()
# y = df["nondurable"].to_numpy()
# z = df["durable"].to_numpy()

# dwt_levels, x_coeffs = dwt.run_dwt(x, MOTHER)
# _, y_coeffs = dwt.run_dwt(y, MOTHER, dwt_levels)
# _, z_coeffs = dwt.run_dwt(z, MOTHER, dwt_levels)

# # %%
# # * Plot each series component separately
# fig, ax = plt.subplots(dwt_levels, 1)
# components = {}
# for l in range(1, dwt_levels + 1):
#     print(l)
#     components[l] = {}
#     for c, c_coeffs in zip(["x", "y", "z"], [x_coeffs, y_coeffs, z_coeffs]):
#         components[l][c] = dwt.reconstruct_signal_component(c_coeffs, MOTHER, l)
#         ax[l - 1].plot(components[l][c])


# # %% [markdown]
# # # ## Simple linear regression
# # ### Nondurables consumption
# # %%
# results_nondur = simple_regression(df, "expectation", "nondurable")
# results_nondur.summary()

# # %% [markdown]
# # ### Wavelet approximation

# # %%
# t = df["date"].to_numpy()
# x = df["expectation"].to_numpy()
# smooth_x, dwt_levels = dwt.smooth_signal(x, MOTHER)
# y = df["nondurable"].to_numpy()

# # %%
# # * Plot smoothing
# fig1 = dwt.plot_smoothing(
#     smooth_x, t, x, name="Actual", figsize=(10, 10), ascending=True
# )
# plt.xlabel("Date")

# # %%
# # * Compare components of both x and y
# smooth_y, _ = dwt.smooth_signal(y, MOTHER)

# fig2, ax = plt.subplots(dwt_levels, 1, sharex=True)

# for l in range(1, dwt_levels + 1):
#     ax[l - 1].plot(t, smooth_x[l]["signal"], label="expectation", color="k")
#     ax[l - 1].plot(t, smooth_y[l]["signal"], label="nondurable", color="r")


# # %%
# approximations = wavelet_approximation(
#     smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels
# )

# # %%
# # * Remove D_1 and D_2
# apprx = approximations[2]
# apprx.summary()

# # %%
# # * Run time scale regression
# tscale_results = time_scale_regression(x_coeffs, y_coeffs, dwt_levels, MOTHER)
# tscale_results

# # %% [markdown]
# # ### Durables consumption

# # %%
# results_dur = simple_regression(df, "expectation", "durable")
# results_dur.summary()

# # %% [markdown]
# # ### Wavelet approximation

# # %%
# t = df["date"].to_numpy()
# x = df["expectation"].to_numpy()
# smooth_x, dwt_levels = dwt.smooth_signal(x, MOTHER)
# y = df["durable"].to_numpy()

# # %%
# # * Plot smoothing
# fig1 = dwt.plot_smoothing(smooth_x, t, x, name="Expectations", figsize=(10, 10))
# plt.xlabel("Date")
# # fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
# fig1.tight_layout()

# # %%
# # * Compare components of both x and y
# smooth_y, _ = dwt.smooth_signal(y, MOTHER)

# fig2, ax = plt.subplots(dwt_levels, 1, sharex=True)

# for l in range(1, dwt_levels + 1):
#     ax[l - 1].plot(t, smooth_x[l]["signal"], label="expectation", color="k")
#     ax[l - 1].plot(t, smooth_y[l]["signal"], label="durable", color="r")


# # %%
# approximations = wavelet_approximation(
#     smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels
# )

# # %%
# # * Remove D_1 through D_5
# apprx = approximations[5]
# apprx.summary()

# # %%
# # * Run time scale regression
# tscale_results = time_scale_regression(x_coeffs, z_coeffs, dwt_levels, MOTHER)
# tscale_results


# # %% [markdown]
# # # # French data for comparison to Andrade et al. (2023)
# # print(
# #     """Conversely, Andrade et al. (2023) find a positive relationship between
# # inflation expectations and durables consumption among French households.
# # Individuals expecting positive inflation are between 1.277% and 1.721% more
# # likely to report having made a durables purchase in the past 12 months. Similarly,
# # individuals expecting positive inflation are between 0.055% and 0.839% more
# # likely to report the present being the “right time to purchase” durables. Further,
# # they also find that the relationships hold strictly for qualitative inflation
# # expectations (i.e. whether inflation will increase, decrease, or stay the same),
# # not for the quantitative estimates individuals provide. France’s statistic agency,
# # the Institut National de la Statistique et des Études Économiques (INSEE), asks
# # respondents for both qualitative and quantitative expectations, so we can test
# # both phenomena."""
# # )

# # # %% [markdown]
# # # ## Get data

# # # %%
# # # * Inflation expectations (qualitative)
# # raw_data = rd.get_insee_data("000857180")
# # inf_exp, _, _ = rd.clean_insee_data(raw_data)
# # ## Rename value column
# # inf_exp.rename(columns={"value": "expectation"}, inplace=True)
# # print("Descriptive stats for inflation expectations")
# # print(inf_exp.describe())

# # # %%
# # # * Durables consumption
# # raw_data = rd.get_insee_data("000857181")
# # dur_consump, _, _ = rd.clean_insee_data(raw_data)
# # dur_consump.rename(columns={"value": "durable"}, inplace=True)
# # print("Descriptive stats for personal durables consumption")
# # print(dur_consump.describe())

# # # %%
# # # * Merge dataframes to remove extra dates
# # df = inf_exp.merge(dur_consump, how="left")
# # df.dropna(inplace=True)
# # print(
# #     f"""Inflation expectations observations: {len(inf_exp)},\nDurables
# #       consumption observations: {len(dur_consump)}.\nNew dataframe lengths: {len(df)}"""
# # )
# # print(df.head(), "\n", df.tail())

# # # %%
# # sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

# # # %% [markdown]
# # # ## Simple linear regression
# # # ### Durables consumption

# # # %%
# # results_dur = simple_regression(df, "expectation", "durable")
# # results_dur.summary()

# # # %% [markdown]
# # # ### Wavelet approximation

# # # %%
# # MOTHER = "db8"
# # t = df["date"].to_numpy()
# # x = df["expectation"].to_numpy()
# # smooth_x, dwt_levels = dwt.smooth_signal(x, MOTHER)
# # y = df["durable"].to_numpy()

# # # %%
# # # * Plot smoothing
# # fig1 = dwt.plot_smoothing(
# #     smooth_x, t, x, name="Expectations", figsize=(10, 10), ascending=True
# # )
# # plt.xlabel("Date")
# # fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
# # fig1.tight_layout()

# # # %%
# # approximations = wavelet_approximation(
# #     smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels
# # )

# # # %%
# # # * Remove detail components 1-5
# # apprx = approximations[5]
# # apprx.summary()

# # # %%
# # # * Plot series
# # fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# # plot_fit(results_dur, exog_idx="expectation", ax=ax[0])
# # plot_fit(apprx, exog_idx=1, ax=ax[1])

# # # %%
# # # * Inflation expectations (quantitative)
# # raw_data = pd.read_csv("../data/inf.csv", delimiter=";")
# # print(raw_data.info())
# # raw_data.head()
# # # %%
# # # * Clean data
# # inf_exp = raw_data.copy()
# # ## Rename value column
# # inf_exp.rename(
# #     columns={
# #         "Unnamed: 0": "date",
# #         "Average Inflation Perception": "perception",
# #         "Average Inflation Expectation": "expectation",
# #     },
# #     inplace=True,
# # )
# # inf_exp["date"] = pd.to_datetime(inf_exp["date"], dayfirst=True)
# # print("Descriptive stats for inflation expectations")
# # print(inf_exp.describe())

# # # %%
# # # * Durables consumption
# # raw_data = rd.get_insee_data("000857181")
# # dur_consump, _, _ = rd.clean_insee_data(raw_data)
# # dur_consump.rename(columns={"value": "durable"}, inplace=True)
# # print("Descriptive stats for personal durables consumption")
# # print(dur_consump.describe())

# # # %%
# # # * Merge dataframes to remove extra dates
# # df = inf_exp.merge(dur_consump, how="left")
# # df.dropna(inplace=True)
# # print(
# #     f"""Inflation expectations observations: {len(inf_exp)},\nDurables
# #       consumption observations: {len(dur_consump)}.\nNew dataframe lengths: {len(df)}"""
# # )
# # print(df.head(), "\n", df.tail())

# # # %%
# # sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

# # # %% [markdown]
# # # ## Simple linear regression
# # # ### Durables consumption

# # # %%
# # results_dur = simple_regression(df, "expectation", "durable")
# # results_dur.summary()

# # # %% [markdown]
# # # ### Wavelet approximation

# # # %%
# # MOTHER = "db8"
# # t = df["date"].to_numpy()
# # x = df["expectation"].to_numpy()
# # smooth_x, dwt_levels = dwt.smooth_signal(x, MOTHER)
# # y = df["durable"].to_numpy()

# # # %%
# # # * Plot smoothing
# # fig1 = dwt.plot_smoothing(
# #     smooth_x, t, x, name="Expectations", figsize=(10, 10), ascending=True
# # )
# # plt.xlabel("Date")
# # fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
# # fig1.tight_layout()

# # # %%
# # approximations = wavelet_approximation(
# #     smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels, verbose=True
# # )

# # # %%
# # # * Remove detail components D_1 and D_2
# # apprx = approximations[2]
# # apprx.summary()

# # # %%
# # # * Plot series
# # fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# # plot_fit(results_dur, exog_idx="expectation", ax=ax[0])
# # plot_fit(apprx, exog_idx=1, ax=ax[1])


# # %%
def main() -> None:
    """Run script"""
    # * Inflation expectations
    raw_data = rd.get_fed_data("MICH", units="pc1")
    inf_exp, _, _ = rd.clean_fed_data(raw_data)
    inf_exp.rename(columns={"value": "expectation"}, inplace=True)
    print("Descriptive stats for inflation expectations")
    print(inf_exp.describe())

    # * Non-durables consumption, monthly
    raw_data = rd.get_fed_data("PCEND", units="pc1")
    nondur_consump, _, _ = rd.clean_fed_data(raw_data)
    nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)
    print("Descriptive stats for personal non-durables consumption")
    print(nondur_consump.describe())

    # * Durables consumption, monthly
    raw_data = rd.get_fed_data("PCEDG", units="pc1")
    dur_consump, _, _ = rd.clean_fed_data(raw_data)
    dur_consump.rename(columns={"value": "durable"}, inplace=True)
    print("Descriptive stats for personal durables consumption")
    print(dur_consump.describe())

    # * Personal savings rate
    raw_data = rd.get_fed_data("PSAVERT", units="pc1")
    save, _, _ = rd.clean_fed_data(raw_data)
    save.rename(columns={"value": "savings"}, inplace=True)
    print("Descriptive stats for personal savings rate")
    print(save.describe())

    # * Merge dataframes to align dates and remove extras
    df = inf_exp.merge(nondur_consump, how="left")
    df = df.merge(dur_consump, how="left")
    df = df.merge(save, how="left")
    print(
        f"""Inflation expectations observations: {len(inf_exp)}, \n
        Non-durables consumption observations: {len(nondur_consump)}, \n
        Durables consumption observations: {len(dur_consump)}, \n
        Savings observation {len(save)}.\nNew dataframe lengths: {len(df)}"""
    )
    print(df.head(), "\n", df.tail())
    print("--------------------------Descriptive stats--------------------------\n")
    print(df.describe())

    sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

    # * Wavelet decomposition
    t = df["date"].to_numpy()

    ## Create objects for DWT
    exp_dwt = dwt.DataForDWT(df["expectation"].to_numpy(), MOTHER)
    nondur_consump_dwt = dwt.DataForDWT(df["nondurable"].to_numpy(), MOTHER)
    dur_consump_dwt = dwt.DataForDWT(df["durable"].to_numpy(), MOTHER)
    save_dwt = dwt.DataForDWT(df["savings"].to_numpy(), MOTHER)

    logger.debug("exp mother wavelet %s", type(exp_dwt.mother_wavelet))
    results_exp_dwt = dwt.run_dwt(exp_dwt)
    results_nondur_consump_dwt = dwt.run_dwt(nondur_consump_dwt)
    results_dur_consump_dwt = dwt.run_dwt(dur_consump_dwt)
    results_save_dwt = dwt.run_dwt(save_dwt)

    df_melt = pd.melt(df, ["date"])
    df_melt.rename(columns={"value": "%"}, inplace=True)

    # * Plot log-by-log change amount frequency
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")
    sns.kdeplot(data=df_melt, x="%", hue="variable", ax=ax)

    # * Plot each series component separately
    _ = plot_compare_components(
        "expectation",
        "nondurable",
        results_exp_dwt.coeffs,
        results_nondur_consump_dwt.coeffs,
        t,
        results_exp_dwt.levels,
        MOTHER,
        figsize=(15, 10),
    )

    _ = plot_compare_components(
        "expectation",
        "durable",
        results_exp_dwt.coeffs,
        results_dur_consump_dwt.coeffs,
        t,
        results_exp_dwt.levels,
        MOTHER,
        figsize=(15, 10),
    )

    _ = plot_compare_components(
        "expectation",
        "savings",
        results_exp_dwt.coeffs,
        results_save_dwt.coeffs,
        t,
        results_exp_dwt.levels,
        MOTHER,
        figsize=(15, 10),
    )

    # * Plot initial series
    _, (bx) = plt.subplots(1, 1)
    bx = sns.lineplot(data=df_melt, x="date", y="%", hue="variable", ax=bx)

    plt.show()


if __name__ == "__main__":
    main()

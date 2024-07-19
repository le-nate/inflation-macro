"""Plot results from wavelet transformations"""

# %%
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pywt
import seaborn as sns

from constants import ids
from src import descriptive_stats, dwt, process_camme, regression
from src import helpers
from src.logging_helpers import define_other_module_log_level
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("Warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ! Define mother wavelet
MOTHER = "db4"
mother_wavelet = pywt.Wavelet(MOTHER)

# * Define constant currency years
CONSTANT_DOLLAR_DATE = "2017-12-01"

# * Define statistical tests to run on data
STATISTICS_TESTS = [
    "count",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "Jarque-Bera",
    "Shapiro-Wilk",
    "Ljung-Box",
]
HYPOTHESIS_THRESHOLD = [0.1, 0.05, 0.001]


# %% [markdown]
## Pre-process data
# US data

#  %%
# * CPI
raw_data = retrieve_data.get_fed_data(ids.US_CPI)
cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
cpi.rename(columns={"value": "cpi"}, inplace=True)

# * Inflation rate
raw_data = retrieve_data.get_fed_data(ids.US_CPI, units="pc1", freq="m")
measured_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
measured_inf.rename(columns={"value": "inflation"}, inplace=True)

# * Inflation expectations
raw_data = retrieve_data.get_fed_data(ids.US_INF_EXPECTATIONS)
inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
inf_exp.rename(columns={"value": "expectation"}, inplace=True)

# * Non-durables consumption, monthly
raw_data = retrieve_data.get_fed_data(ids.US_NONDURABLES_CONSUMPTION)
nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

# * Durables consumption, monthly
raw_data = retrieve_data.get_fed_data(ids.US_DURABLES_CONSUMPTION)
dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
dur_consump.rename(columns={"value": "durable"}, inplace=True)

# * Personal savings
raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS)
save, _, _ = retrieve_data.clean_fed_data(raw_data)
save.rename(columns={"value": "savings"}, inplace=True)

# * Personal savings rate
raw_data = retrieve_data.get_fed_data(ids.US_SAVINGS_RATE)
save_rate, _, _ = retrieve_data.clean_fed_data(raw_data)
save_rate.rename(columns={"value": "savings_rate"}, inplace=True)

# * Merge dataframes to align dates and remove extras
us_data = cpi.merge(measured_inf, how="left")
us_data = us_data.merge(inf_exp, how="left")
us_data = us_data.merge(nondur_consump, how="left")
us_data = us_data.merge(dur_consump, how="left")
us_data = us_data.merge(save, how="left")
us_data = us_data.merge(save_rate, how="left")

# * Remove rows without data for all measures
us_data.dropna(inplace=True)

# * Add real value columns
logger.info(
    "Using constant dollars from %s, CPI: %s",
    CONSTANT_DOLLAR_DATE,
    us_data[us_data["date"] == pd.Timestamp(CONSTANT_DOLLAR_DATE)]["cpi"].iat[0],
)
us_data = helpers.add_real_value_columns(
    data=us_data,
    nominal_columns=["nondurable", "durable", "savings"],
    cpi_column="cpi",
    constant_date=CONSTANT_DOLLAR_DATE,
)
us_data = helpers.calculate_diff_in_log(
    data=us_data, columns=["cpi", "real_nondurable", "real_durable", "real_savings"]
)

usa_sliced = pd.concat([us_data.head(), us_data.tail()])
usa_sliced

# %% [markdown]
# French data
# %%
# * CPI
raw_data = retrieve_data.get_fed_data(ids.FR_CPI)
fr_cpi, _, _ = retrieve_data.clean_fed_data(raw_data)
fr_cpi.rename(columns={"value": "cpi"}, inplace=True)

# * Measured inflation
raw_data = retrieve_data.get_fed_data(ids.FR_CPI, units="pc1", freq="m")
fr_inf, _, _ = retrieve_data.clean_fed_data(raw_data)
fr_inf.rename(columns={"value": "inflation"}, inplace=True)

# * Inflation expectations
_, fr_exp = process_camme.preprocess(process_camme.camme_dir)
## Remove random lines with month as letter
fr_exp = fr_exp[fr_exp["month"].apply(isinstance, args=(int,))]
## Create date column
fr_exp["date"] = pd.to_datetime(fr_exp[["year", "month"]].assign(DAY=1))
## Use just quantitative expectations and date
fr_exp = fr_exp[["date", "inf_exp_val_inc", "inf_exp_val_dec"]]
## Convert to negative for averaging
fr_exp["inf_exp_val_dec"] = fr_exp["inf_exp_val_dec"] * -1
## Melt then pivot to get average expectation for each month
fr_exp_melt = pd.melt(fr_exp, ["date"])
fr_exp = pd.pivot_table(fr_exp_melt, index="date", aggfunc="mean")
fr_exp.rename(columns={"value": "expectation"}, inplace=True)

# * Food consumption
raw_data = retrieve_data.get_insee_data(ids.FR_FOOD_CONSUMPTION)
fr_food_cons, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_food_cons.rename(columns={"value": "food"}, inplace=True)

# * Goods consumption
raw_data = retrieve_data.get_insee_data(ids.FR_GOODS_CONSUMPTION)
fr_goods_cons, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_goods_cons.rename(columns={"value": "goods"}, inplace=True)

# * Durables consumption
raw_data = retrieve_data.get_insee_data(ids.FR_DURABLES_CONSUMPTION)
fr_dur_cons, _, _ = retrieve_data.clean_insee_data(raw_data)
fr_dur_cons.rename(columns={"value": "durables"}, inplace=True)

# %%
fr_data = fr_cpi.merge(fr_inf, how="left")
fr_data = fr_data.merge(fr_exp.reset_index(), how="left")
fr_data = fr_data.merge(fr_food_cons, how="left")
fr_data = fr_data.merge(fr_goods_cons, how="left")
fr_data = fr_data.merge(fr_dur_cons, how="left")

fr_sliced = pd.concat([fr_data.head(), fr_data.tail()])
fr_sliced

# %%
# * Create measured inflation dataframe
inf_data = pd.merge(
    us_data[["date", "inflation", "expectation"]],
    fr_data[["date", "inflation", "expectation"]],
    on="date",
    suffixes=("_us", "_fr"),
)

inf_data.columns = [
    "Date",
    "Measured (US)",
    "Expectations (US)",
    "Measured (France)",
    "Expectations (France)",
]

# %%
inf_melt = pd.melt(inf_data, ["Date"])
inf_melt.rename(columns={"value": "Measured (%)"}, inplace=True)

# %% [markdown]
##### Figure XX - Time series: Measured Inflation (US and France)
# %%
_, (ax, bx) = plt.subplots(2, 1, sharex=True)

# * US subplot
measures_to_plot = ["Measured (US)", "Expectations (US)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
ax = sns.lineplot(data=data, x="Date", y="Measured (%)", hue="variable", ax=ax)
ax.legend().set_title(None)

# * French subplot
measures_to_plot = ["Measured (France)", "Expectations (France)"]
data = inf_melt[inf_melt["variable"].isin(measures_to_plot)]
bx = sns.lineplot(data=data, x="Date", y="Measured (%)", hue="variable", ax=bx)
bx.legend().set_title(None)
plt.suptitle("Inflation Rates, US and France")
plt.tight_layout()

# %%[markdown]
## Table 1 Descriptive statistics

# %%
usa_melt = pd.melt(us_data, ["date"])
usa_melt.rename(columns={"value": "Billions ($)"}, inplace=True)

# %% [markdown]
##### Figure 2 - Time series: Inflation Expectations, Nondurables Consumption, Durables Consumption, and Savings (US)
# %%
_, (bx) = plt.subplots(1, 1)
measures_to_plot = ["nondurable", "durable"]
data = usa_melt[usa_melt["variable"].isin(measures_to_plot)]
bx = sns.lineplot(data=data, x="date", y="Billions ($)", hue="variable", ax=bx)
plt.title("Real consumption levels, United States (2017 dollars)")

# %% [markdown]
##### Figure 3 - Distribution of Inflation Expectations, Nondurables Consumption, Durables Consumption, and Savings (US)
# %%
plot_columns = [
    "inflation",
    "expectation",
    "diff_log_real_nondurable",
    "diff_log_real_durable",
    "diff_log_real_savings",
]
sns.pairplot(us_data[plot_columns], corner=True, kind="reg", plot_kws={"ci": None})

# %% [markdown]
##### Table 1: Descriptive statistics
# %%
descriptive_statistics_results = descriptive_stats.generate_descriptive_statistics(
    us_data, STATISTICS_TESTS, export_table=False
)
descriptive_statistics_results

# %% [markdown]
##### Table XX: Correlation matrix
# %%
us_corr = descriptive_stats.correlation_matrix_pvalues(
    data=us_data[[c for c in us_data.columns if "log" in c]],
    hypothesis_threshold=HYPOTHESIS_THRESHOLD,
    decimals=2,
    display=False,
    export_table=False,
)
us_corr

# %% [markdown]
##### Figure XX - Time series: Inflation Expectations, Food Consumption, Durables Consumption (France)
# %%
fr_melt = pd.melt(fr_data, ["date"])
fr_melt.rename(columns={"value": "Billions (€)"}, inplace=True)

fig, (ax, bx) = plt.subplots(1, 2)
measures_to_plot = ["food", "durables"]
data = fr_melt[fr_melt["variable"].isin(measures_to_plot)]
ax = sns.lineplot(data=data, x="date", y="Billions (€)", hue="variable", ax=ax)
plt.title("Consumption levels, France")

# %% [markdown]
##### Figure XX - Distribution of Inflation Expectations, Nondurables Consumption, Durables Consumption (France)
# %%
sns.pairplot(fr_data, corner=True, kind="reg", plot_kws={"ci": None})

# %% [markdown]
##### Table 1: Descriptive statistics
# %%
fr_data.describe()

# %% [markdown]
## 3.2) Exploratory analysis

# %% [markdown]
### 3.2.1) Time scale decomposition

# %%
# * Create data objects for each measure
exp_for_dwt = dwt.DataForDWT(us_data["expectation"].to_numpy(), mother_wavelet)
nondur_for_dwt = dwt.DataForDWT(us_data["nondurable"].to_numpy(), mother_wavelet)
dur_for_dwt = dwt.DataForDWT(us_data["durable"].to_numpy(), mother_wavelet)
save_for_dwt = dwt.DataForDWT(us_data["savings"].to_numpy(), mother_wavelet)

# * Run DWTs and extract smooth signals
results_exp_dwt = dwt.smooth_signal(exp_for_dwt)
results_nondur_dwt = dwt.smooth_signal(nondur_for_dwt)
results_dur_dwt = dwt.smooth_signal(dur_for_dwt)
results_save_dwt = dwt.smooth_signal(save_for_dwt)

# * Numpy array for date
t = us_data["date"].to_numpy()

# %% [markdown]
# Figure 4 - Time scale decomposition of expectations and nondurables consumption (US)
# %%
# * Plot comparison decompositions of expectations and other measure
_ = regression.plot_compare_components(
    a_label="expectation",
    b_label="nondurable",
    smooth_a_coeffs=results_exp_dwt.coeffs,
    smooth_b_coeffs=results_nondur_dwt.coeffs,
    time=t,
    levels=results_exp_dwt.levels,
    wavelet=MOTHER,
    figsize=(15, 10),
)

# %% [markdown]
# Figure 5 - Time scale decomposition of expectations and durables consumption (US)
# %%
_ = regression.plot_compare_components(
    a_label="expectation",
    b_label="durable",
    smooth_a_coeffs=results_exp_dwt.coeffs,
    smooth_b_coeffs=results_dur_dwt.coeffs,
    time=t,
    levels=results_exp_dwt.levels,
    wavelet=MOTHER,
    figsize=(15, 10),
)

# %% [markdown]
# Figure XX - Time scale decomposition of expectations and savings (US)
# %%
_ = regression.plot_compare_components(
    a_label="expectation",
    b_label="savings",
    smooth_a_coeffs=results_exp_dwt.coeffs,
    smooth_b_coeffs=results_save_dwt.coeffs,
    time=t,
    levels=results_exp_dwt.levels,
    wavelet=MOTHER,
    figsize=(15, 10),
)

# %% [markdown]
### 3.2.2) Individual time series: Continuous wavelet transforms
# TODO CWTs for exp, nondur, dur, save

# %% [markdown]
### 3.2.3) Time series co-movements: Cross wavelet transforms and phase difference
# TODO Figure 9 - Key, Phase Difference
# TODO XWTs for exp with nondur, dur, and save

# %% [markdown]
## 3.3) Regression analysis
### 3.3.1) Baseline model
# Nondurables consumption
# %%
results_nondur = regression.simple_regression(us_data, "expectation", "nondurable")
results_nondur.summary()

# %% [markdown]
# Durables consumption
# %%
results_dur = regression.simple_regression(us_data, "expectation", "durable")
results_dur.summary()

# %% [markdown]
# Savings
# %%
results_dur = regression.simple_regression(us_data, "expectation", "savings")
results_dur.summary()

# %% [markdown]
## 3.3.2) Wavelet approximation
# Figure 12 - Wavelet Smoothing of Inflation Expectations (US)

# %%
fig, title = dwt.plot_smoothing(
    results_exp_dwt.smoothed_signal_dict,
    t,
    exp_for_dwt.y_values,
    figsize=(10, 10),
)
plt.xlabel("Year")
plt.ylabel(f"{title.capitalize()}")
fig.tight_layout()
plt.show()


# %% [markdown]
# Table 4 - Wavelet Approximation: OLS Regression Inflation Expectations and
# Nondurables Consumption (US) <br><br>

# For our wavelet approximation of the OLS regression of nondurables consumption
# on inflation expectations, we use S_2, removing D_1 and D_2. Table 4 shows
# the results. Overall, there is little change in the results compared to the
# simple regression.

# %%
approximations = regression.wavelet_approximation(
    smooth_t_dict=results_exp_dwt.smoothed_signal_dict,
    original_y=nondur_for_dwt.y_values,
    levels=results_exp_dwt.levels,
)

# * Remove D_1 and D_2
apprx = approximations[2]
apprx.summary()

# %% [markdown]
# Table 5 - Wavelet Approximation: OLS Regression Inflation Expectations and
# Durables Consumption with S_2 (US) <br><br>

# The same is true for durables, when removing components D_1 and D_2 (Table 5).
# Given how absolutely inconclusive the OLS regression is, we further test the
# impact of a regression with almost purely smoothed expectations in S_5=S_6+D_6
# as well. Again, we cannot reject the null hypothesis (Table 6).

# %%
approximations = regression.wavelet_approximation(
    smooth_t_dict=results_exp_dwt.smoothed_signal_dict,
    original_y=dur_for_dwt.y_values,
    levels=results_exp_dwt.levels,
)

# * Remove D_1 and D_2
apprx = approximations[2]
apprx.summary()

# %% [markdown]
# Table 6 - Wavelet Approximation: OLS Regression Inflation Expectations and
# Durables Consumption with S_5 (US) <br><br>

# %%
# * Remove D_1 through D_5
apprx = approximations[5]
apprx.summary()

# %% [markdown]
# Table XX - Wavelet Approximation: OLS Regression Inflation Expectations and
# Savings (US) <br><br>

# %%
approximations = regression.wavelet_approximation(
    smooth_t_dict=results_exp_dwt.smoothed_signal_dict,
    original_y=save_for_dwt.y_values,
    levels=results_exp_dwt.levels,
)

# * Remove D_1 and D_2
apprx = approximations[2]
apprx.summary()


# %% [markdown]
## 3.3) Time scale regression
# Table 7 - Time Scale Decomposition: OLS Regression of Nondurables Consumption
# on Inflation Expectations (US)

# %%
time_scale_results = regression.time_scale_regression(
    input_coeffs=results_exp_dwt.coeffs,
    output_coeffs=results_nondur_dwt.coeffs,
    levels=results_exp_dwt.levels,
    mother_wavelet=MOTHER,
)
time_scale_results

# %% [markdown]
# Table 8 - Time Scale Decomposition: OLS Regression of Durables Consumption on
# Inflation Expectations (US)

# %%
time_scale_results = regression.time_scale_regression(
    input_coeffs=results_exp_dwt.coeffs,
    output_coeffs=results_dur_dwt.coeffs,
    levels=results_exp_dwt.levels,
    mother_wavelet=MOTHER,
)
time_scale_results


# %% [markdown]
# Table XX - Time Scale Decomposition: OLS Regression of Savings on
# Inflation Expectations (US)

# %%
time_scale_results = regression.time_scale_regression(
    input_coeffs=results_exp_dwt.coeffs,
    output_coeffs=results_save_dwt.coeffs,
    levels=results_exp_dwt.levels,
    mother_wavelet=MOTHER,
)
time_scale_results


# %% [markdown]
# Figure 13 - Example, Phase differences
# %%
# TODO

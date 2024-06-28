"""Plot results from wavelet transformations"""

# %%
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import pywt
import seaborn as sns

from src import cwt, dwt, regression, xwt
from src.helpers import define_other_module_log_level
from src import retrieve_data

# * Logging settings
logger = logging.getLogger(__name__)
define_other_module_log_level("debug")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# ! Define mother wavelet
MOTHER = "db4"
mother_wavelet = pywt.Wavelet(MOTHER)


# %% [markdown]
## Pre-process data

#  %%
# * Inflation expectations
raw_data = retrieve_data.get_fed_data("MICH")
inf_exp, _, _ = retrieve_data.clean_fed_data(raw_data)
inf_exp.rename(columns={"value": "expectation"}, inplace=True)
print("Descriptive stats for inflation expectations")
print(inf_exp.describe())

# * Inflation expectations (percent change)
raw_data = retrieve_data.get_fed_data("MICH", units="pc1")
inf_exp_perc, _, _ = retrieve_data.clean_fed_data(raw_data)
inf_exp_perc.rename(columns={"value": "expectation_%_chg"}, inplace=True)
print("Descriptive stats for inflation expectations (percent change)")
print(inf_exp_perc.describe())

# * Non-durables consumption, monthly
raw_data = retrieve_data.get_fed_data("PCEND", units="pc1")
nondur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)

# * Durables consumption, monthly
raw_data = retrieve_data.get_fed_data("PCEDG", units="pc1")
dur_consump, _, _ = retrieve_data.clean_fed_data(raw_data)
dur_consump.rename(columns={"value": "durable"}, inplace=True)

# * Personal savings rate
raw_data = retrieve_data.get_fed_data("PSAVERT")  # , units="pc1")
save, _, _ = retrieve_data.clean_fed_data(raw_data)
save.rename(columns={"value": "savings"}, inplace=True)

# * Merge dataframes to align dates and remove extras
df = inf_exp.merge(inf_exp_perc, how="left")
df = df.merge(nondur_consump, how="left")
df = df.merge(dur_consump, how="left")
df = df.merge(save, how="left")

df_sliced = pd.concat([df.head(), df.tail()])
df_sliced

# %%[markdown]
## Descriptive statistics

# %%
df_melt = pd.melt(df[[c for c in df.columns if "%_chg" not in c]], ["date"])
df_melt.rename(columns={"value": "%"}, inplace=True)

# %% [markdown]
##### Figure 2 - Time series: Inflation Expectations, Nondurables Consumption, Durables Consumption, and Savings (US)
_, (bx) = plt.subplots(1, 1)
bx = sns.lineplot(data=df_melt, x="date", y="%", hue="variable", ax=bx)

# %% [markdown]
##### Figure 3 - Distribution of Inflation Expectations, Nondurables Consumption, Durables Consumption, and Savings (US)
# %%
sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

# %% [markdown]
##### Table 1: Descriptive statistics
# %%
df.describe()

# %% [markdown]
## 3.2) Exploratory analysis

# %% [markdown]
### 3.2.1) Time scale decomposition

# %%
# * Create data objects for each measure
exp_for_dwt = dwt.DataForDWT(df["expectation"].to_numpy(), mother_wavelet)
nondur_for_dwt = dwt.DataForDWT(df["nondurable"].to_numpy(), mother_wavelet)
dur_for_dwt = dwt.DataForDWT(df["durable"].to_numpy(), mother_wavelet)
save_for_dwt = dwt.DataForDWT(df["savings"].to_numpy(), mother_wavelet)

# * Run DWTs and extract smooth signals
results_exp_dwt = dwt.smooth_signal(exp_for_dwt)
results_nondur_dwt = dwt.smooth_signal(nondur_for_dwt)
results_dur_dwt = dwt.smooth_signal(dur_for_dwt)
results_save_dwt = dwt.smooth_signal(save_for_dwt)

# * Numpy array for date
t = df["date"].to_numpy()

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
results_nondur = regression.simple_regression(df, "expectation", "nondurable")
results_nondur.summary()

# %% [markdown]
# Durables consumption
# %%
results_dur = regression.simple_regression(df, "expectation", "durable")
results_dur.summary()

# %% [markdown]
# Savings
# %%
results_dur = regression.simple_regression(df, "expectation", "savings")
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

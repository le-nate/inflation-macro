"""Conduct regression using denoised data via DWT"""

# %%
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_fit

from scripts import dwt
from analysis import retrieve_data as rd


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
    smooth_x_dict: dict,
    original_y: npt.NDArray,
    levels: int,
    add_constant: bool = True,
    verbose: bool = False,
) -> dict:
    """Regresses smooth components"""
    regressions_dict = {}
    crystals = list(range(1, levels + 1))
    for c in crystals:
        x_c = smooth_x_dict[c]["signal"]
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


# %% [markdown]
# # US data for comparison to Coibion et al. (2021)
print(
    """Coibion et al. (2021) find that inflation expectations have a positive
relationship with nondurable and services consumption and a negative relationship
with durable consumption. A 1% increase in inflation expectations correlates with
a 1.8% increase in nondurables and services consumption and 1.5% decrease in
durables consumption."""
)

# %% [markdown]
# ## Get data

# %%
# * Inflation expectations
raw_data = rd.get_fed_data("MICH")
inf_exp, _, _ = rd.clean_fed_data(raw_data)
## Rename value column
inf_exp.rename(columns={"value": "expectation"}, inplace=True)
print("Descriptive stats for inflation expectations")
print(inf_exp.describe())

# %%
# * Non-durables consumption, monthly
raw_data = rd.get_fed_data("PCEND", units="pc1")
nondur_consump, _, _ = rd.clean_fed_data(raw_data)
nondur_consump.rename(columns={"value": "nondurable"}, inplace=True)
print("Descriptive stats for personal non-durables consumption")
print(nondur_consump.describe())

# %%
# * Durables consumption, monthly
raw_data = rd.get_fed_data("PCEDG", units="pc1")
dur_consump, _, _ = rd.clean_fed_data(raw_data)
dur_consump.rename(columns={"value": "durable"}, inplace=True)
print("Descriptive stats for personal durables consumption")
print(dur_consump.describe())

# %%
# * Merge dataframes to remove extra dates
df = inf_exp.merge(nondur_consump, how="left")
df = df.merge(dur_consump, how="left")
print(
    f"""Inflation expectations observations: {len(inf_exp)}, \nNon-durables 
      consumption observations: {len(nondur_consump)}, \nDurables 
      consumption observations: {len(dur_consump)}.\nNew dataframe lengths: {len(df)}"""
)
print(df.head(), "\n", df.tail())

# %%
sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

# %% [markdown]
# ## Simple linear regression
# ### Nondurables consumption
# %%
results_nondur = simple_regression(df, "expectation", "nondurable")
results_nondur.summary()

# %% [markdown]
# ### Wavelet approximation

# %%
mother = "db4"
t = df["date"].to_numpy()
x = df["expectation"].to_numpy()
smooth_x, dwt_levels = dwt.smooth_signal(x, mother)
y = df["nondurable"].to_numpy()

# %%
# * Plot smoothing
fig1 = dwt.plot_smoothing(smooth_x, t, x, name="Expectations", figsize=(10, 10))
plt.xlabel("Date")
fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
fig1.tight_layout()

# %%
approximations = wavelet_approximation(
    smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels
)

# %%
# * Remove D_1 and D_2
apprx = approximations[2]

# %%
# * Plot series
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

plot_fit(results_nondur, exog_idx="expectation", ax=ax[0])
plot_fit(apprx, exog_idx=1, ax=ax[1])

# %% [markdown]
# ### Durables consumption

# %%
results_dur = simple_regression(df, "expectation", "durable")
results_dur.summary()

# %% [markdown]
# ### Wavelet approximation

# %%
mother = "db4"
t = df["date"].to_numpy()
x = df["expectation"].to_numpy()
smooth_x, dwt_levels = dwt.smooth_signal(x, mother)
y = df["durable"].to_numpy()

# %%
# * Plot smoothing
fig1 = dwt.plot_smoothing(smooth_x, t, x, name="Expectations", figsize=(10, 10))
plt.xlabel("Date")
fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
fig1.tight_layout()

# %%
approximations = wavelet_approximation(
    smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels, verbose=True
)

# %%
# * Remove D_1 through D_5
apprx = approximations[5]

# %%
# * Plot series
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

plot_fit(results_dur, exog_idx="expectation", ax=ax[0])
plot_fit(apprx, exog_idx=1, ax=ax[1])


# %% [markdown]
# # French data for comparison to Andrade et al. (2023)
print(
    """Conversely, Andrade et al. (2023) find a positive relationship between
inflation expectations and durables consumption among French households.
Individuals expecting positive inflation are between 1.277% and 1.721% more
likely to report having made a durables purchase in the past 12 months. Similarly, 
individuals expecting positive inflation are between 0.055% and 0.839% more 
likely to report the present being the “right time to purchase” durables. Further,
they also find that the relationships hold strictly for qualitative inflation
expectations (i.e. whether inflation will increase, decrease, or stay the same),
not for the quantitative estimates individuals provide. France’s statistic agency,
the Institut National de la Statistique et des Études Économiques (INSEE), asks
respondents for both qualitative and quantitative expectations, so we can test
both phenomena."""
)

# %% [markdown]
# ## Get data

# %%
# * Inflation expectations
raw_data = rd.get_insee_data("000857180")
inf_exp, _, _ = rd.clean_insee_data(raw_data)
## Rename value column
inf_exp.rename(columns={"value": "expectation"}, inplace=True)
print("Descriptive stats for inflation expectations")
print(inf_exp.describe())

# %%
# * Durables consumption
raw_data = rd.get_insee_data("000857181")
dur_consump, _, _ = rd.clean_insee_data(raw_data)
dur_consump.rename(columns={"value": "durable"}, inplace=True)
print("Descriptive stats for personal durables consumption")
print(dur_consump.describe())

# %%
# * Merge dataframes to remove extra dates
df = inf_exp.merge(dur_consump, how="left")
df.dropna(inplace=True)
print(
    f"""Inflation expectations observations: {len(inf_exp)},\nDurables 
      consumption observations: {len(dur_consump)}.\nNew dataframe lengths: {len(df)}"""
)
print(df.head(), "\n", df.tail())

# %%
sns.pairplot(df, corner=True, kind="reg", plot_kws={"ci": None})

# %% [markdown]
# ## Simple linear regression
# ### Durables consumption

# %%
results_dur = simple_regression(df, "expectation", "durable")
results_dur.summary()

# %% [markdown]
# ### Wavelet approximation

# %%
mother = "db4"
t = df["date"].to_numpy()
x = df["expectation"].to_numpy()
smooth_x, dwt_levels = dwt.smooth_signal(x, mother)
y = df["durable"].to_numpy()

# %%
# * Plot smoothing
fig1 = dwt.plot_smoothing(
    smooth_x, t, x, name="Expectations", figsize=(10, 10), ascending=True
)
plt.xlabel("Date")
fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
fig1.tight_layout()

# %%
approximations = wavelet_approximation(
    smooth_x_dict=smooth_x, original_y=y, levels=dwt_levels, verbose=True
)

# %%
# * Remove all detail components
apprx = approximations[5]

# %%
# * Plot series
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

plot_fit(results_dur, exog_idx="expectation", ax=ax[0])
plot_fit(apprx, exog_idx=1, ax=ax[1])


# %%
def main() -> None:
    """Run script"""


if __name__ == "__main__":
    main()

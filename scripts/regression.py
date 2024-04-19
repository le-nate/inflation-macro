"""Conduct regression using denoised data via DWT"""

# %%
import matplotlib.pyplot as plt
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
    x = data[x_var]
    y = data[y_var]
    if add_constant:
        x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    return results


def time_scale_regression(
    smooth_x: dict,
    smooth_y: dict,
    levels: int,
    add_constant: bool = True,
) -> dict:
    """Regresses smooth components"""
    regressions_dict = {}
    crystals = list(range(1, levels + 1))
    for c in crystals:
        regressions_dict[c] = {}
        x_c = smooth_x[c]["signal"]
        y_c = smooth_y[c]["signal"]
        if add_constant:
            x_c = sm.add_constant(x_c)
        model = sm.OLS(y_c, x_c)
        results_smooth = model.fit()
        print("\n\n")
        print(f"-----Smoothed model, X1a_er - D_{list(range(1, c+1))}-----")
        print("\n")
        print(results_smooth.summary())
        regressions_dict[c]["y_c"] = y_c
        regressions_dict[c]["results"] = results_smooth
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

# %%
# * Plot series
plot_fit(results_nondur, exog_idx="expectation")

# %% [markdown]
# ### Wavelet approximation

# %%
mother = "db4"
t = df["date"].to_numpy()
x = df["expectation"].to_numpy()
smooth_x, dwt_levels = dwt.smooth_signal(x, mother)
y = df["nondurable"].to_numpy()
smooth_y, _ = dwt.smooth_signal(x, mother, dwt_levels)

# %%
# * Plot smoothing
fig1 = dwt.plot_smoothing(smooth_x, t, x, name="Expectations", figsize=(10, 10))
plt.xlabel("Date")
fig1.suptitle(f"Wavelet smoothing of Expectations (J={dwt_levels})")
fig1.tight_layout()

fig2 = dwt.plot_smoothing(smooth_y, t, y, name="Expectations", figsize=(10, 10))
plt.xlabel("Date")
fig2.suptitle(f"Wavelet smoothing of Nondurables consumption (J={dwt_levels})")
fig2.tight_layout()

# %%
regressions = time_scale_regression(
    smooth_x=smooth_x, smooth_y=smooth_y, levels=dwt_levels
)

# %% [markdown]
# ### Durables consumption

# %%
results_dur = simple_regression(df, "expectation", "durable")
results_dur.summary()

# %%
# * Plot series
plot_fit(results_dur, exog_idx="expectation")

# %% [markdown]
# ## Wavelet approximations


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

# %%
# * Plot series
plot_fit(results_dur, exog_idx="expectation")


# %%
def main() -> None:
    """Run script"""


if __name__ == "__main__":
    main()

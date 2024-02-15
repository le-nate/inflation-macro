"""
Perform wavelet approximation through decomposition
See Ramsey, J. B., Gallegati, M., Gallegati, M., & Semmler, W. (2010). 
        Instrumental variables and wavelet decompositions. Economic Modelling,
        27(6), 1498–1513. https://doi.org/10.1016/j.econmod.2010.07.011
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt

from analysis import retrieve_data as rd

## Matplotlib Settings

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=BIGGER_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %%
raw_data = rd.get_fed_data("PSAVERT")

# %%
## Convert to dataframe
df = pd.DataFrame(raw_data)
print(df.info(verbose=True), "\n")
df.tail()
# %%
## Convert dtypes
df.replace(".", np.nan, inplace=True)
df.dropna(inplace=True)
df["value"] = pd.to_numeric(df["value"])
df["date"] = pd.to_datetime(df["date"])

## Drop extra columns
df = df[["date", "value"]]
print(df.dtypes)
print(df.describe(), "\n")
df.head()
# %%
# df2 = pd.DataFrame(inf)
# print(df2.info(verbose=True), "\n")
# df2.head()
# %%
## Convert dtypes
# df2.replace(".", np.nan, inplace=True)
# df2.dropna(inplace=True)
# # df2["value2"] = df2["value"].astype(float)
# # df2["date"] = pd.to_datetime(df2["date"])

## Drop extra columns
# # df2 = df2[["date", "value2"]]
# print(df2.dtypes)
# df2.head()
# %%
## Merge dataframes
# df = df.merge(df2, how="left")
# print(df.info())
# df.tail()

# %%
# Perform wavelet decomposition
## Choose the wavelet type, here using Daubechies 4
WAVELET = "db4"
w = pywt.Wavelet(WAVELET)
## Choose the maximum decomposition level
level = pywt.dwt_max_level(data_len=len(df), filter_len=w.dec_len)
print("Max decomposition level:", level)

t = df["date"]
y = df["value"]

coeffs = pywt.wavedec(y, WAVELET, level=level)

## Input name of time series
print("Enter name of time series (to be included in plot)")
name = input()

## Plot original time series and smoothed and detailed coefficients
plt.figure(figsize=(10, 10))
plt.plot(t, y, "b", label=name.title())

# Plot approximation coefficients
for i in range(1, level + 1):
    approx_coefficients = pywt.upcoef("a", coeffs[i], WAVELET, level=i, take=len(y))
    plt.plot(
        t[: len(approx_coefficients)],
        approx_coefficients,
        "r--",
        label=f"Approximation Coefficients {i}",
    )

# Plot detail coefficients
for i in range(1, level + 1):
    detail_coefficients = pywt.upcoef("d", coeffs[i], WAVELET, level=i, take=len(y))
    plt.plot(
        t[: len(detail_coefficients)],
        detail_coefficients,
        "g--",
        label=f"Detail Coefficients {i}",
    )

plt.xlabel("Year")
plt.ylabel("Value")
plt.title(f"Wavelet decomposition of {name.lower()}")
plt.legend(loc="best")
plt.grid(True)
plt.show()

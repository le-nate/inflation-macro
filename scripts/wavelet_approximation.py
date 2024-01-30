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

expinf = rd.get_fed_data("EXPINF1YR")
inf = rd.get_fed_data("CPIAUCSL", units="pc1", freq="m")

# %%
## Convert to dataframe
df = pd.DataFrame(expinf)
print(df.info(verbose=True), "\n")
df.tail()
# %%
## Convert dtypes
df["value"] = pd.to_numeric(df["value"])
df["date"] = pd.to_datetime(df["date"])

## Drop extra columns
df = df[["date", "value"]]
print(df.dtypes)
df.head()
# %%
df2 = pd.DataFrame(inf)
print(df2.info(verbose=True), "\n")
df2.head()
# %%
## Convert dtypes
df2.replace(".", np.nan, inplace=True)
df2.dropna(inplace=True)
df2["value2"] = df2["value"].astype(float)
df2["date"] = pd.to_datetime(df2["date"])

## Drop extra columns
df2 = df2[["date", "value2"]]
print(df2.dtypes)
df2.head()
# %%
## Merge dataframes
df = df.merge(df2, how="left")
print(df.info())
df.tail()

# %%
# Perform wavelet decomposition
WAVELET = "db4"  # Choose the wavelet type, here using Daubechies 4
LEVEL = 3  # Choose the decomposition level

t = df["date"]
y = df["value"]
y_inf = df["value2"]

coeffs = pywt.wavedec(y, WAVELET, level=LEVEL)

# Plot the original signal and the approximation and detail coefficients
plt.figure(figsize=(10, 6))
plt.plot(t, y, "b", label="Expected Inflation")
plt.plot(t, y_inf, "y", label="Measured Inflation")

# Plot approximation coefficients
for i in range(1, LEVEL + 1):
    approx_coefficients = pywt.upcoef("a", coeffs[i], WAVELET, level=i, take=len(y))
    plt.plot(
        t[: len(approx_coefficients)],
        approx_coefficients,
        "r--",
        label=f"Approximation Coefficients {i}",
    )

# Plot detail coefficients
for i in range(1, LEVEL + 1):
    detail_coefficients = pywt.upcoef("d", coeffs[i], WAVELET, level=i, take=len(y))
    plt.plot(
        t[: len(detail_coefficients)],
        detail_coefficients,
        "g--",
        label=f"Detail Coefficients {i}",
    )

plt.xlabel("i")
plt.ylabel("Value")
plt.title("Wavelet Decomposition of Expected Inflation")
plt.legend(loc="best")
plt.grid(True)
plt.show()

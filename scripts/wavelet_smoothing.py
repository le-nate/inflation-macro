"""
Smoothing of signals via wavelet reconstruction
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
# raw_data = rd.get_fed_data("CPIAUCSL", units="pc1", freq="m")
# t, y = rd.clean_fed_data(raw_data)

raw_data = rd.get_insee_data("000857179")
t, y = rd.clean_insee_data(raw_data)

# %%
## Define the wavelet type
WAVELET = "db4"
w = pywt.Wavelet(WAVELET)

## Choose the maximum decomposition level
levels = pywt.dwt_max_level(data_len=len(y), filter_len=w.dec_len)
print(f"Max decomposition level of {levels} for time series length of {len(y)}")

coeffs = pywt.wavedec(y, WAVELET, level=levels)

# %%
## Create dict for reconstructed signals
smooth_signals = {}

## Time series with uneven result in mismatched lengths with the reconstructed
## signal, so we remove a value from the approximated signal
if len(y) % 2 != 0:
    print(f"Odd number of observations dectected (Length: {len(y)}). Trim data? (y/n)")
    trim = input()
else:
    trim = "n"

if trim == "y":
    print("Trim beginning or end of time series? (begin/end)")
    trim = input()

## Loop through levels and add detail level components
for l in range(levels):
    smooth_coeffs = coeffs.copy()
    smooth_signals[l] = {}
    ## Set remaining detail coefficients to zero
    for i in range(1, len(smooth_coeffs) - l):
        smooth_coeffs[i] = np.zeros_like(smooth_coeffs[i])
    smooth_signals[l]["coeffs"] = smooth_coeffs
    # Reconstruct the signal using only the approximation coefficients
    reconst = pywt.waverec(smooth_coeffs, WAVELET)
    if trim == "begin":
        smooth_signals[l]["signal"] = reconst[1:]
    elif trim == "end":
        smooth_signals[l]["signal"] = reconst[:-1]
    else:
        smooth_signals[l]["signal"] = reconst

# %%

## Input name of time series
print("Enter name of time series (to be included in plot)")
name = input()

fig = plt.figure(figsize=(10, 10))
## Loop through levels and add detail level components
for i, (level, signal) in enumerate(smooth_signals.items(), 1):
    plt.subplot(len(smooth_signals), 1, i)  # Create a subplot for each smooth signal
    plt.plot(t, y, label=name.title())
    plt.plot(t, signal["signal"])
    plt.xlabel("Year")
    plt.grid()
    plt.title(rf"Approximation: $S_{{j-{level}}}$")
    plt.legend()

plt.xlabel("Year")
plt.ylabel(f"{name.capitalize()}")
fig.suptitle(f"Wavelet smoothing of {name.lower()}")
fig.tight_layout()  # Adjust layout to prevent overlapping subplots
fig.show()

# %%

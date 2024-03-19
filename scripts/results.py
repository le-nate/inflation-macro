"""Plot results from wavelet transformations"""

# %%
import matplotlib.pyplot as plt
import pywt

from wavelet_smoothing import smooth_signal
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

raw_data = rd.get_insee_data("000857179")
t, y = rd.clean_insee_data(raw_data)

## Define the wavelet type
WAVELET = "db4"
w = pywt.Wavelet(WAVELET)

## Choose the maximum decomposition level
levels = pywt.dwt_max_level(data_len=len(y), filter_len=w.dec_len)
print(f"Max decomposition level of {levels} for time series length of {len(y)}")

coeffs = pywt.wavedec(y, WAVELET, level=levels)

smooth_signals = smooth_signal(coeffs, levels, WAVELET)

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

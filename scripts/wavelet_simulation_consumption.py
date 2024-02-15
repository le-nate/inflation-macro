"""
Wavelet decomposition of consumption simulation, from:
Ramsey, J. B., Gallegati, M., Gallegati, M., & Semmler, W. (2010). 
        Instrumental variables and wavelet decompositions. Economic Modelling,
        27(6), 1498–1513. https://doi.org/10.1016/j.econmod.2010.07.011
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pywt

from simulation_consumption import consumption

# %%
# Generate values for i
i_values = np.linspace(1, 512, 1000)

# Calculate consumption values
consumption_values = consumption(i_values)

# %%
# Perform wavelet decomposition
WAVELET = "sym12"  # Choose the wavelet type, here using Daubechies 4
w = pywt.Wavelet(WAVELET)
## Choose the maximum decomposition level
level = pywt.dwt_max_level(data_len=len(consumption_values), filter_len=w.dec_len)
print("Max decomposition level:", level)

coeffs = pywt.wavedec(consumption_values, WAVELET, level=level)

# %%
# Plot the original signal and the approximation and detail coefficients
plt.figure(figsize=(10, 6))
plt.plot(i_values, consumption_values, "b", label="Original Signal")

# Plot approximation coefficients
for i in range(1, level + 1):
    approx_coefficients = pywt.upcoef(
        "a", coeffs[i], WAVELET, level=i, take=len(consumption_values)
    )
    plt.plot(
        i_values[: len(approx_coefficients)],
        approx_coefficients,
        "r--",
        label=f"Approximation Coefficients {i}",
    )

# Plot detail coefficients
for i in range(1, level + 1):
    detail_coefficients = pywt.upcoef(
        "d", coeffs[i], WAVELET, level=i, take=len(consumption_values)
    )
    plt.plot(
        i_values[: len(detail_coefficients)],
        detail_coefficients,
        "g--",
        label=f"Detail Coefficients {i}",
    )

plt.xlabel("i")
plt.ylabel("Value")
plt.title("Wavelet Decomposition of Consumption Function")
plt.legend(loc="best")
plt.grid(True)
plt.show()

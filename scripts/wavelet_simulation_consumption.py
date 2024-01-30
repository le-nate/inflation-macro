import numpy as np
import matplotlib.pyplot as plt
import pywt

from simulation_consumption import consumption

# Generate values for i
i_values = np.linspace(1, 512, 1000)

# Calculate consumption values
consumption_values = consumption(i_values)

# Perform wavelet decomposition
WAVELET = "db4"  # Choose the wavelet type, here using Daubechies 4
LEVEL = 3  # Choose the decomposition level

coeffs = pywt.wavedec(consumption_values, WAVELET, level=LEVEL)

# Plot the original signal and the approximation and detail coefficients
plt.figure(figsize=(10, 6))
plt.plot(i_values, consumption_values, "b", label="Original Signal")

# Plot approximation coefficients
for i in range(1, LEVEL + 1):
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
for i in range(1, LEVEL + 1):
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

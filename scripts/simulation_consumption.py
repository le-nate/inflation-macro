"""Simple cyclical income and consumption functions, per
Ramsey, J. B., Gallegati, M., Gallegati, M., & Semmler, W. (2010). Instrumental 
variables and wavelet decompositions. Economic Modelling, 27(6), 1498–1513. 
https://doi.org/10.1016/j.econmod.2010.07.011"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

AUTHOR = "Nathaniel Lawrence"


## Income functions
def y_exp(i):
    return np.exp(2 * (i / 512))


def y_sin(i):
    return np.sin(i * (np.pi / 7.8))


def income(i):
    return y_exp(i) + y_sin(i)


def error(i):
    """error = N(0, sigma^2), where sigma = 0.3 * MAD[y_sin]"""
    return


def error_function(i):
    # Define the sine function
    sin_function = y_sin(i)

    # Calculate the mean absolute difference
    mean_abs_diff_squared = np.mean(np.abs(sin_function - np.mean(sin_function)) ** 2)

    # Generate a random sample from a normal distribution with mean 0
    normal_sample = norm.rvs(loc=0, scale=1, size=len(i))

    # Multiply the normal sample by the square root of the mean absolute difference squared
    error_term = np.sqrt(mean_abs_diff_squared) * normal_sample

    return error_term


def consumption(i, exp_const=0.9, sin_const=0.6):
    return exp_const * y_exp(i) + sin_const * y_sin(i) + error_function(i)


# Example usage:
x_values = np.linspace(0, 2 * np.pi, 1000)
error_terms = error_function(x_values)

# Generate values for i
i_values = np.linspace(1, 512, 1000)

# Calculate function values for each i
income_values = income(i_values)
consumption_values = consumption(i_values)

# Plot the graph
plt.plot(
    i_values,
    income_values,
    label=r"Income = $e^{2(i/512)} + \sin\left(\frac{i \pi}{7.8}\right)$",
)
plt.plot(
    i_values,
    consumption_values,
    label=r"Consumption = 0.9*$e^{2(i/512)} + 0.6*\sin\left(\frac{i \pi}{7.8}\right) + \epsilon$",
)
plt.xlabel("i")
plt.ylabel("Income and consumption")
plt.title("Cyclical income and consumption functions")
plt.legend()
plt.grid(True)
plt.show()

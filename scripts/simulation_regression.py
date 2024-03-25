"""Regress simulated data, including with wavelet approximations"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_fit

from simulation_consumption import consumption
import wavelet_smoothing

# %%
## Unobservable model
ALPHA_0 = 10
ALPHA_1 = 0.6
ALPHA_2 = 0.9
MEAN_U = 0
SIGMA2_U = 0.6
MEAN_V = 0
SIGMA2_V = 0.3


def x_1ai(i: float) -> float:
    """Calculate x_1ai independent value"""
    return 2 * np.sin(i * (np.pi / 64))


def x_2i(i: float) -> float:
    """Calculate x_2i independent value"""
    return np.exp(2 * (i / 512))


def error_term(i: float, mean: float, stdev: float) -> float:
    """Calculate u_i or v_i error terms"""
    return norm.rvs(loc=mean, scale=stdev, size=len(i))


def x_1a_er(i: float) -> float:
    """Observable error term"""
    return x_1ai(i) + error_term(i, MEAN_V, SIGMA2_V)


def unobservable_model(i: float) -> list:
    """Unobservable model: y_ai = ALPHA_0 + ALPHA_1*x_1ai + ALPHA_2*x_2i + u_i"""
    return (
        ALPHA_0
        + ALPHA_1 * x_1ai(i)
        + ALPHA_2 * x_2i(i)
        + error_term(i, MEAN_U, SIGMA2_U)
    )


def error_model(i: float) -> list:
    """Model used for regression with variable error
    y_ai = ALPHA_0 + ALPHA_1*x_1ai_er + ALPHA_2*x_2i + u_i - ALPHA_1*v_i"""
    return (
        ALPHA_0
        + ALPHA_1 * x_1a_er(i)
        + ALPHA_2 * x_2i(i)
        + error_term(i, MEAN_U, SIGMA2_U)
        - ALPHA_1 * error_term(i, MEAN_V, SIGMA2_V)
    )


def main() -> None:
    """Run script"""
    i_values = np.linspace(1, 512, 512)
    raw_data = unobservable_model(i_values)
    x1 = x_1ai(i_values)
    x2 = x_2i(i_values)
    i = sm.add_constant(np.column_stack((x1, x2)))
    y_ai = raw_data
    model = sm.OLS(y_ai, i)
    results = model.fit()
    print(results.summary())

    # Plot the graph
    plot_fit(results, exog_idx=2)
    plt.plot(x2, y_ai, label="y_ai")
    plt.show()


if __name__ == "__main__":
    main()

"""Regress simulated data, including with mother approximations"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pywt
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_fit

from simulation_consumption import consumption
import scripts.dwt as dwt

# %%
## Unobservable model
ALPHA_0 = 10
ALPHA_1 = 0.6
ALPHA_2 = 0.9
MEAN_U = 0
SIGMA2_U = 0.6
MEAN_V = 0
SIGMA2_V = 0.3


def x_1ai(i: npt.NDArray) -> float:
    """Calculate x_1ai independent value"""
    return 2 * np.sin(i * (np.pi / 64))


def x_2i(i: npt.NDArray) -> float:
    """Calculate x_2i independent value"""
    return np.exp(2 * (i / 512))


def error_term(i: npt.NDArray, mean: float, stdev: float) -> npt.NDArray:
    """Calculate u_i or v_i error terms"""
    return norm.rvs(loc=mean, scale=stdev, size=len(i))


def x_1a_er(i: npt.NDArray, error: npt.NDArray) -> npt.NDArray:
    """Observable error term"""
    return x_1ai(i) + error


def actual_model(
    x_1a: npt.NDArray, x_2: npt.NDArray, error: npt.NDArray
) -> npt.NDArray:
    """(Actual) Unobservable model:
    y_ai = ALPHA_0 + ALPHA_1*x_1ai + ALPHA_2*x_2i + u_i"""
    return ALPHA_0 + ALPHA_1 * x_1a + ALPHA_2 * x_2 + error


def error_model(
    x1_er: npt.NDArray, x2: npt.NDArray, error: npt.NDArray, error2: npt.NDArray
) -> npt.NDArray:
    """Model used for regression with error in variable x_1a:
    y_ai = ALPHA_0 + ALPHA_1*x_1ai_er + ALPHA_2*x_2i + u_i - ALPHA_1*v_i"""
    return ALPHA_0 + ALPHA_1 * x1_er + ALPHA_2 * x2 + error - ALPHA_1 * error2


def main() -> None:
    """Run script"""
    i_values = np.linspace(1, 512, 512)
    x1 = x_1ai(i_values)
    x2 = x_2i(i_values)
    error_u = error_term(i_values, MEAN_U, SIGMA2_U)
    error_v = error_term(i_values, MEAN_V, SIGMA2_V)
    x1_er = x_1a_er(i_values, error_v)

    # * Regress actual model
    y_ai = actual_model(x1, x2, error_u)
    i = sm.add_constant(np.column_stack((x1, x2)))
    model = sm.OLS(y_ai, i)
    results_actual = model.fit()

    # * Regress error model
    y_ai_er = error_model(x1_er, x2, error_u, error_v)
    i = sm.add_constant(np.column_stack((x1_er, x2)))
    model = sm.OLS(y_ai_er, i)
    results_error = model.fit()

    # * Calculate differences between actual model and error model coefficients
    for i, j in enumerate(["alpha0", "alpha1", "alpha2"]):
        print(
            f"""\n-----% of stderr for {j}: {(
                results_actual.params[i] - results_error.params[i]
            )/ results_actual.bse[i]}-----\n"""
        )

    # * Denoise error model with cumulative subtraction of D1, D2, and D3 crystals
    mother = "sym12"
    levels = 6
    smooth_signals, dwt_levels = dwt.smooth_signal(x1_er, mother, levels=levels)

    # * Regress smoothed model
    regressions_dict = {}
    crystals = [1, 2, 3]
    for c in crystals:
        print(c)
        regressions_dict[c] = {}
        x = smooth_signals[c]["signal"]
        y = error_model(x, x2, error_u, error_v)
        i = sm.add_constant(np.column_stack((x, x2)))
        model = sm.OLS(y, i)
        results_smooth = model.fit()
        print("\n\n")
        print(f"-----Smoothed model, X1a_er - D_{[i for i in range(1, c+1)]}-----")
        print("\n")
        print(results_smooth.summary())
        regressions_dict[c]["y"] = y
        regressions_dict[c]["results"] = results_smooth

    # * Plot smoothing
    fig1 = dwt.plot_smoothing(
        smooth_signals, x=i_values, y=x1_er, name="Error model", figsize=(10, 10)
    )
    print(smooth_signals.keys())
    plt.xlabel("i")
    plt.ylabel("Error model")
    fig1.suptitle(f"Wavelet smoothing of Error model (J={dwt_levels})")
    fig1.tight_layout()

    # * Define subplots
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # * Plot regression of actual model
    plot_fit(results_actual, exog_idx=2, ax=ax1)
    ax1.plot(x2, y_ai, label="y_ai")
    ax1.set_title(
        rf"Unobservable model, $\alpha_{{1}}$: {results_actual.params[1]}, $\sigma^{{2}}$: {results_actual.bse[1]}"
    )

    # * Plot regression of error model
    plot_fit(results_error, exog_idx=2, ax=ax2)
    ax2.plot(x2, y_ai_er, label="y_ai_er", color="orange")
    ax2.set_title(
        rf"Error model, $\alpha_{{1}}$: {results_error.params[1]}, $\sigma^{{2}}$: {results_error.bse[1]}"
    )

    # TODO Plot regression of smoothed model
    plot_fit(regressions_dict[3]["results"], exog_idx=2, ax=ax3)
    ax3.plot(x2, regressions_dict[3]["y"], label=r"y($S_{{3}}$)", color="pink")
    ax3.set_title(
        rf"""$S_{{3}}$ model, $\alpha_{{1}}$: {regressions_dict[3]["results"].params[1]}, $\sigma^{{2}}$: {regressions_dict[3]["results"].bse[1]}"""
    )

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

"""Cross-project helper functions"""

import logging
from typing import Dict, Generator, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def define_other_module_log_level(level: str) -> None:
    """Disable logger ouputs for other modules up to defined `level`"""
    for log_name in logging.Logger.manager.loggerDict:
        if log_name != "__name__":
            log_level = getattr(logging, level.upper())
            logging.getLogger(log_name).setLevel(log_level)


def nested_dict_values(nested_dict: Dict) -> Generator[any, any, any]:
    """Extract nested dict values"""
    for v in nested_dict.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def nested_list_values(nested_list: List[List[str]]) -> Generator[any, any, any]:
    """Extract nested list values"""
    for v in nested_list:
        if isinstance(v, list):
            yield from nested_list_values(v)
        else:
            yield v


def plot_signficance_levels(
    ax: plt.Axes,
    signficance_levels: npt.NDArray,
    t_values: npt.NDArray,
    period: npt.NDArray,
    **kwargs
) -> None:
    """Plot contours for 95% significance level\n
    **kwargs**\n
    `sig_colors=`: 'k'\n
    `sig_linewidths=`: 2"""
    extent = [t_values.min(), t_values.max(), 0, max(period)]
    ax.contour(
        t_values,
        np.log2(period),
        signficance_levels,
        [-99, 1],
        colors=kwargs["colors"],
        linewidths=kwargs["linewidths"],
        extent=extent,
    )


def plot_cone_of_influence(
    ax: plt.Axes,
    coi: npt.NDArray,
    t_values: npt.NDArray,
    levels: List[float],
    period: npt.NDArray,
    dt: float,
    tranform_type: str,
    **kwargs
) -> None:
    """Plot shaded area for cone of influence, where edge effects may occur\n
    **Params**\n
    `transform_type=`: "cwt" or "xwt"\n
    `color=`: 'k'
    `alpha =`: 0.3\n
    `hatch =`: "--"
    """
    color = kwargs["color"]
    alpha = kwargs["alpha"]
    hatch = kwargs["hatch"]
    t_array = np.concatenate(
        [
            t_values,
            t_values[-1:] + dt,
            t_values[-1:] + dt,
            t_values[:1] - dt,
            t_values[:1] - dt,
        ]
    )
    if tranform_type == "cwt":
        coi_array = np.concatenate(
            [
                np.log2(coi),
                [levels[2]],
                np.log2(period[-1:]),
                np.log2(period[-1:]),
                [levels[2]],
            ]
        ).clip(
            min=-2.5
        )  # ! To keep cone of influence from bleeding off graph
    if tranform_type == "xwt":
        coi_array = coi
    ax.fill(
        t_array,
        coi_array,
        color,
        alpha=alpha,
        hatch=hatch,
    )

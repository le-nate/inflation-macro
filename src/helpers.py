"""Cross-project helper functions"""

from typing import Dict, Generator, List

import numpy as np
import pandas as pd

from src import retrieve_data


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


def add_real_value_columns(
    data: pd.DataFrame, nominal_columns: List[str], **kwargs
) -> pd.DataFrame:
    """Convert nominal to real values for each column in list"""
    for col in nominal_columns:
        data[f"real_{col}"] = retrieve_data.convert_column_to_real_value(
            data=data, column=col, **kwargs
        )
    return data


def calculate_diff_in_log(
    data: pd.DataFrame, columns: List[str], new_columns: bool = True
) -> pd.DataFrame:
    """
    Calculate rolling difference of log(values), adding as new column if defined
    """
    for col in columns:
        if new_columns:
            col_name = f"diff_log_{col}"
        else:
            col_name = col
        data[col_name] = 100 * np.log(data[col]).diff()
    return data

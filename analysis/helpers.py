"""Cross-project helper functions"""

import logging
from typing import Dict, Generator, List


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

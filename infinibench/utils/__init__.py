#!/usr/bin/env python3
"""Utilities package for InfiniBench."""

from infinibench.utils.input_loader import load_input_file, load_inputs_from_paths
from infinibench.utils.metrics import Metric, ScalarMetric, TimeseriesMetric
from infinibench.utils.time_utils import get_timestamp

__all__ = [
    "load_input_file",
    "load_inputs_from_paths",
    "Metric",
    "ScalarMetric",
    "TimeseriesMetric",
    "get_timestamp",
]

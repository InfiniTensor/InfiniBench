#!/usr/bin/env python3
"""Operators package for InfiniMetrics."""

from infinimetrics.operators.flops_calculator import (
    FLOPSCalculator,
    calculate_bandwidth,
)

__all__ = [
    "FLOPSCalculator",
    "calculate_bandwidth",
]

try:
    from infinimetrics.operators.infinicore_adapter import InfiniCoreAdapter
    __all__.append("InfiniCoreAdapter")
except ImportError:
    pass

try:
    from infinimetrics.operators.infiniops_adapter import InfiniOpsAdapter
    __all__.append("InfiniOpsAdapter")
except ImportError:
    pass

#!/usr/bin/env python3
"""Operators package for InfiniBench."""

from infinibench.operators.flops_calculator import (
    FLOPSCalculator,
    calculate_bandwidth,
)
from infinibench.operators.infinicore_adapter import InfiniCoreAdapter

__all__ = [
    "FLOPSCalculator",
    "calculate_bandwidth",
    "InfiniCoreAdapter",
    "InfiniOpsAdapter",
]


def __getattr__(name):
    """Load InfiniOps only when callers explicitly request it."""
    if name == "InfiniOpsAdapter":
        from infinibench.operators.infiniops_adapter import InfiniOpsAdapter

        return InfiniOpsAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

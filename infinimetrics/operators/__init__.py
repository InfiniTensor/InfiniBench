#!/usr/bin/env python3
"""Operators package for InfiniMetrics."""

from infinimetrics.operators.flops_calculator import (
    FLOPSCalculator,
    calculate_bandwidth,
)

__all__ = [
    "FLOPSCalculator",
    "calculate_bandwidth",
    "InfiniCoreAdapter",
    "InfiniOpsAdapter",
]


def __getattr__(name):
    """Load adapters only when callers explicitly request them."""
    if name == "InfiniCoreAdapter":
        from infinimetrics.operators.infinicore_adapter import InfiniCoreAdapter

        return InfiniCoreAdapter
    if name == "InfiniOpsAdapter":
        from infinimetrics.operators.infiniops_adapter import InfiniOpsAdapter

        return InfiniOpsAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

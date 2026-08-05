#!/usr/bin/env python3
"""Operators package for InfiniBench."""

from infinibench.operators.flops_calculator import (
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
    """Load optional operator adapters only when explicitly requested."""
    if name == "InfiniCoreAdapter":
        from infinibench.operators.infinicore_adapter import InfiniCoreAdapter

        return InfiniCoreAdapter
    if name == "InfiniOpsAdapter":
        from infinibench.operators.infiniops_adapter import InfiniOpsAdapter

        return InfiniOpsAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

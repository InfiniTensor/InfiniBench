#!/usr/bin/env python3
"""
infinimetrics inference package
Unified Reasoning Evaluation Framework
"""

__version__ = "1.0.0"
__author__ = "Infinimetrics Team"

from .infer_config import (
    InferConfig, InferConfigManager,
    InferMode, FrameworkType,
    DirectInferArgs, ServiceInferArgs
)
from .infer_runner_base import (
    InferRunnerBase, BenchmarkResult,
    Metric, ScalarMetric, TimeseriesMetric
)
from .adapter_base import InferAdapter
from .infer_runner_factory import InferRunnerFactory

__all__ = [
    # config
    "InferConfig", "InferConfigManager",
    "InferMode", "FrameworkType",
    "DirectInferArgs", "ServiceInferArgs",
    
    # Runner
    "InferRunnerBase", "BenchmarkResult",
    "Metric", "ScalarMetric", "TimeseriesMetric",
    
    # adapter
    "InferAdapter",
    
    # factory
    "InferRunnerFactory",
]

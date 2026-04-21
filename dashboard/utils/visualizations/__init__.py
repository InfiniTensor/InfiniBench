#!/usr/bin/env python3
"""Visualization functions for InfiniBench dashboard.

This package provides visualization utilities organized by test type:
- base: Common/legacy visualization functions
- hardware: Hardware test visualizations (memory sweep, cache bandwidth)
- communication: Communication test visualizations
- inference: Inference test visualizations
- operator: Operator test visualizations
- training: Training test visualizations
- summary_tables: Summary tables for different test types
"""

# Base functions (common)
from .base import (
    create_gauge_chart,
    plot_timeseries_auto,
)

# Communication functions
from .communication import (
    plot_metric_vs_size,
    plot_comparison_matrix,
)

# Inference functions
from .inference import (
    render_inference_metrics,
    render_memory_gauge,
)

# Summary tables
from .summary_tables import (
    create_summary_table_comm,
    create_summary_table_infer,
    create_summary_table_ops,
)

# Hardware functions
from .hardware import (
    create_summary_table_hw,
    plot_hw_mem_sweep,
    plot_hw_cache,
)

# Operator functions
from .operator import (
    extract_operator_metrics,
    render_operator_performance_charts,
)

# Training functions
from .training import (
    render_performance_curves,
    render_throughput_comparison,
    render_data_tables,
    render_config_details,
)

__all__ = [
    # Base
    "create_gauge_chart",
    "plot_timeseries_auto",
    # Communication
    "plot_metric_vs_size",
    "plot_comparison_matrix",
    # Inference
    "render_inference_metrics",
    "render_memory_gauge",
    # Summary tables
    "create_summary_table_comm",
    "create_summary_table_infer",
    "create_summary_table_ops",
    # Hardware
    "create_summary_table_hw",
    "plot_hw_mem_sweep",
    "plot_hw_cache",
    # Operator
    "extract_operator_metrics",
    "render_operator_performance_charts",
    # Training
    "render_performance_curves",
    "render_throughput_comparison",
    "render_data_tables",
    "render_config_details",
]

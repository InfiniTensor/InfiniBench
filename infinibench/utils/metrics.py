#!/usr/bin/env python3
"""
Metrics module
Defines common metric types used by both inference and training modules
"""

from typing import Dict, Any, Optional


class Metric:
    """Metric base class"""

    def __init__(self, name: str, metric_type: str, unit: Optional[str] = None):
        self.name = name
        self.type = metric_type
        self.unit = unit
        self.value = None
        self.raw_data_url = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"name": self.name, "type": self.type, "unit": self.unit}

        if self.type == "scalar":
            result["value"] = self.value
        elif self.type == "timeseries":
            result["raw_data_url"] = self.raw_data_url

        return result


class ScalarMetric(Metric):
    """Scalar metric"""

    def __init__(self, name: str, value: Any, unit: Optional[str] = None):
        super().__init__(name, "scalar", unit)
        self.value = value


class TimeseriesMetric(Metric):
    """Time-series metric"""

    def __init__(self, name: str, raw_data_url: str, unit: Optional[str] = None):
        super().__init__(name, "timeseries", unit)
        self.raw_data_url = raw_data_url

#!/usr/bin/env python3
"""Test Input Data Classes"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class TestInput:
    """
    Test input data structure.

    Only testcase field is required. Other fields are optional.

    Example:
        >>> input = TestInput(
        ...     testcase="infer.InfiniLM.Direct",
        ...     config={"model": "Qwen3-1.7B"}
        ... )
    """

    testcase: str
    run_id: Optional[str] = None
    time: Optional[str] = None
    success: Optional[int] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate testcase and set default time."""
        if not self.testcase:
            raise ValueError("testcase must be a non-empty string")
        if self.time is None:
            self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None/empty optional fields."""
        result = {"testcase": self.testcase, "config": self.config}
        for key in ["run_id", "time", "success", "metrics"]:
            value = getattr(self, key)
            if value not in (None, []):
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestInput":
        """Create TestInput from dictionary."""
        return cls(
            testcase=data.get("testcase"),
            run_id=data.get("run_id"),
            time=data.get("time"),
            success=data.get("success"),
            config=data.get("config", {}),
            metrics=data.get("metrics", []),
        )

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a value from config with a default."""
        return self.config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """Set a value in config."""
        self.config[key] = value

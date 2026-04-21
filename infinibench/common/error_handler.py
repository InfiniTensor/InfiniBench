#!/usr/bin/env python3
"""Error Handling Utilities for Test Execution."""

import logging
from datetime import datetime
from typing import Any, Dict

from infinibench.common.constants import ErrorCode

logger = logging.getLogger(__name__)

# Memory-related error keywords
MEMORY_KEYWORDS = [
    "out of memory",
    "oom",
    "memory leak",
    "memory allocation failed",
    "insufficient memory",
    "cuda out of memory",
]

# Error logging configuration: error_code -> (is_critical, issue_type, analysis)
_ERROR_LOG_CONFIG = {
    ErrorCode.TIMEOUT: (
        True,
        "timeout",
        "Test timed out. Hardware may be hung or overloaded.",
    ),
    ErrorCode.SYSTEM: (True, "memory", "Memory allocation failed."),
    ErrorCode.CONFIG: (False, "configuration_error", None),
    ErrorCode.GENERIC: (False, "runtime_error", None),
}


class ErrorHandler:
    """Handles error classification and response building."""

    @staticmethod
    def classify_runtime_error(error_msg: str) -> int:
        """
        Classify RuntimeError by analyzing error message.

        Args:
            error_msg: Error message string (lowercase)

        Returns:
            Appropriate error code
        """
        if any(kw in error_msg for kw in MEMORY_KEYWORDS):
            return ErrorCode.SYSTEM
        return ErrorCode.GENERIC

    @staticmethod
    def build_error_response(
        run_id: str,
        testcase: str,
        error_msg: str,
        result_code: int,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a response dict containing error information.

        Args:
            run_id: Test run identifier
            testcase: Test case name
            error_msg: Error message string
            result_code: Error result code
            config: Test configuration

        Returns:
            Dictionary with error details
        """
        # Create cleaned config without injected metadata
        cleaned_config = {
            k: v
            for k, v in config.items()
            if not k.startswith("_")  # Skip _testcase, _run_id, _time
        }

        return {
            "run_id": run_id,
            "testcase": testcase,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result_code": result_code,
            "error_msg": error_msg,
            "success": 1,  # 1 = failure
            "config": cleaned_config,
        }

    @staticmethod
    def log_error(testcase: str, error: Exception, error_code: int) -> None:
        """
        Log error with appropriate severity and context.

        Args:
            testcase: Test case name
            error: Exception instance
            error_code: Error code for classification
        """
        error_msg = str(error)[:300]
        is_critical, issue_type, analysis = _ERROR_LOG_CONFIG.get(
            error_code, (False, "unknown_error", None)
        )

        log_fn = logger.error if is_critical else logger.warning
        prefix = "STABILITY CHECK FAILED" if is_critical else "Test failed"

        lines = [f"Executor: {prefix} for {testcase}", f"  Issue Type: {issue_type}"]
        if is_critical and analysis:
            lines.append("  Severity: CRITICAL")
            lines.append(f"  Analysis: {analysis}")
        lines.append(f"  Error: {error_msg}")

        log_fn("\n".join(lines))

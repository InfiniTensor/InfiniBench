#!/usr/bin/env python3
"""Data utility functions for InfiniBench dashboard."""
import json
import logging

from pathlib import Path
from typing import Any, Dict, List


def load_summary_file(summary_path: str = "../summary_output") -> List[Dict[str, Any]]:
    """Load dispatcher summary files."""

    logger = logging.getLogger(__name__)
    summaries = []
    summary_dir = Path(summary_path)

    if summary_dir.exists():
        for json_file in sorted(
            summary_dir.glob("dispatcher_summary_*.json"), reverse=True
        ):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["file"] = json_file.name
                data["timestamp"] = json_file.stem.replace("dispatcher_summary_", "")
                summaries.append(data)
            except Exception as e:
                logger.warning(f"Failed to load summary {json_file}: {e}")

    return summaries


def normalize_ci_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize CI summary information, prioritizing Git metadata when available.
    """
    # Extract Git and CI environment information
    git_info = data.get("git", {})
    ci_info = data.get("ci_environment", {})

    # Prefer values from git_info if this is a valid Git repository
    if git_info.get("is_git_repo"):
        data["commit"] = git_info.get("commit", "unknown")
        data["short_commit"] = git_info.get("short_commit", "unknown")
        data["branch"] = git_info.get("branch", "unknown")
        data["commit_message"] = git_info.get("commit_message", "unknown")
        data["commit_author"] = git_info.get("commit_author", "unknown")
        data["commit_date"] = git_info.get("commit_date", "unknown")
        data["_has_real_git_info"] = True
    else:
        data["commit"] = git_info.get("commit", "not_in_git_repo")
        data["short_commit"] = git_info.get("short_commit", "not_in_git_repo")
        data["branch"] = git_info.get("branch", "not_in_git_repo")
        data["commit_message"] = git_info.get("commit_message", "Not in Git repository")
        data["_has_real_git_info"] = False

    # Attach CI environment metadata
    if ci_info:
        data["ci_provider"] = ci_info.get("ci_provider", "unknown")
        data["ci_pipeline_id"] = ci_info.get(
            "ci_pipeline_id", ci_info.get("ci_run_id", "")
        )

    # Compute duration if not already present
    if "duration" not in data and "total_duration_seconds" in data:
        data["duration"] = data["total_duration_seconds"]

    # Derive overall CI status
    total = data.get("total_tests", 0)
    failed = data.get("failed_tests", 0)
    if total == 0:
        data["status"] = "无测试"
    elif failed == 0:
        data["status"] = "✅ 成功"
    elif data.get("successful_tests", 0) > 0:
        data["status"] = "⚠️ 部分成功"
    else:
        data["status"] = "❌ 失败"

    return data


def extract_failed_tests_details(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract failed test details from CI summary data.

    Supports two formats:
    1. Directly using the `failed_tests_details` field
    2. Extracting failed cases from `test_results` or `tests` lists
    """
    if "failed_tests_details" in data and data["failed_tests_details"]:
        return data["failed_tests_details"]

    failed_details = []

    # Attempt to extract from test_results list
    if "test_results" in data:
        for test in data["test_results"]:
            if not test.get("success", True):
                failed_details.append(
                    {
                        "test_name": test.get("name", test.get("testcase", "unknown")),
                        "error": test.get(
                            "error", test.get("error_msg", "Unknown error")
                        ),
                        "duration": test.get("duration", 0),
                        "logs": test.get("logs", test.get("output", "")),
                    }
                )

    # Attempt to extract from tests list
    elif "tests" in data:
        for test in data["tests"]:
            if test.get("status") in ["failed", "error"]:
                failed_details.append(
                    {
                        "test_name": test.get("name", test.get("testcase", "unknown")),
                        "error": test.get(
                            "error", test.get("message", "Unknown error")
                        ),
                        "duration": test.get("duration", 0),
                        "logs": test.get("logs", test.get("output", "")),
                    }
                )

    return failed_details


def get_friendly_size(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def extract_accelerator_types(result_json: dict) -> list[str]:
    """Extract the accelerator card type from result_json."""
    types = set()
    try:
        clusters = result_json.get("environment", {}).get("cluster", [])
        for node in clusters:
            accs = node.get("machine", {}).get("accelerators", [])
            for acc in accs:
                if "type" in acc:
                    types.add(acc["type"])
    except Exception:
        pass
    return list(types)


def extract_test_type(testcase: str) -> str:
    """Extract test type from testcase string (e.g., 'comm.nccl.allreduce' -> 'comm')."""
    parts = testcase.split(".")
    return parts[0] if parts else "unknown"


def extract_operation(testcase: str) -> str:
    """Extract operation from testcase string (e.g., 'comm.nccl.allreduce' -> 'allreduce')."""
    parts = testcase.split(".")
    return parts[2] if len(parts) > 2 else "unknown"


def extract_run_info(data: Dict[str, Any], path: Path = None) -> Dict[str, Any]:
    """
    Extract run info from test result data.

    Args:
        data: Test result JSON data
        path: Optional file path (for file-based sources)

    Returns:
        Dictionary with extracted run information
    """
    config = data.get("config", {})
    resolved = data.get("resolved", {})

    device_used = (
        resolved.get("device_used")
        or config.get("device_used")
        or config.get("device_involved", 1)
    )

    nodes = resolved.get("nodes") or data.get("environment", {}).get("cluster_scale", 1)

    result_code = data.get("result_code", 1)
    success = result_code == 0

    metrics = data.get("metrics", [])
    metric_types = [m.get("name", "").split(".")[0] for m in metrics if m.get("name")]

    testcase = data.get("testcase", "")

    return {
        "path": path,
        "testcase": testcase,
        "run_id": data.get("run_id", "unknown"),
        "time": data.get("time", ""),
        "success": success,
        "result_code": result_code,
        "test_type": extract_test_type(testcase),
        "operation": extract_operation(testcase),
        "config": config,
        "resolved": resolved,
        "device_used": device_used,
        "nodes": nodes,
        "metrics_count": len(metrics),
        "metric_types": list(set(metric_types)),
    }

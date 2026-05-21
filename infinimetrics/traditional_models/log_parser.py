#!/usr/bin/env python3
"""Log parser for traditional model training/eval outputs.

Extracts throughput, loss, accuracy, latency, and pass/fail status
from model execution logs.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Common patterns across model frameworks
_PATTERNS = {
    # Loss
    "loss": [
        re.compile(r"(?:loss|Loss)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"Training Loss:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"Epoch \d+.*?loss[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    # Accuracy
    "accuracy": [
        re.compile(
            r"(?:accuracy|acc|Accuracy|Acc)\s*[:=]\s*([0-9]*\.?[0-9]+)",
            re.IGNORECASE,
        ),
        re.compile(r"Top-1 Accuracy:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"mAP:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    # Throughput (samples/sec, images/sec, etc.)
    "throughput": [
        re.compile(
            r"(?:throughput|Throughput|speed|Speed)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(?:samples/s|images/s|items/s|tok/s)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(\d+(?:\.\d+)?)\s*(?:samples/s|images/s|items/s|tok/s)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:samples per second|images per second)\s*[:=]\s*([0-9]*\.?[0-9]+)",
            re.IGNORECASE,
        ),
    ],
    # Latency
    "latency": [
        re.compile(
            r"(?:latency|Latency|time|Time)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(?:ms|s)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:inference time|Inference Time)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(?:ms|s)",
            re.IGNORECASE,
        ),
    ],
    # Epoch
    "epoch": [
        re.compile(r"Epoch\s*\[?(\d+)(?:/(\d+))?\]?", re.IGNORECASE),
    ],
    # Iteration step
    "step": [
        re.compile(
            r"(?:step|Step|iteration|Iteration)\s*\[?(\d+)(?:/(\d+))?\]?",
            re.IGNORECASE,
        ),
    ],
}


def parse_log(log_text: str, model_name: str = "") -> Dict[str, Any]:
    """Parse model training/evaluation log text.

    Args:
        log_text: Full log text from stdout/stderr.
        model_name: Name of the model (for context).

    Returns:
        Dict with parsed metrics:
            - loss: list of float values found
            - accuracy: list of float values found
            - throughput: list of float values found
            - latency: list of float values found
            - final_loss: last loss value or None
            - final_accuracy: last accuracy value or None
            - final_throughput: last throughput value or None
            - final_latency: last latency value or None
            - epochs_seen: int
            - steps_seen: int
            - completed: bool (True if log suggests normal completion)
    """
    result = {
        "loss": [],
        "accuracy": [],
        "throughput": [],
        "latency": [],
        "final_loss": None,
        "final_accuracy": None,
        "final_throughput": None,
        "final_latency": None,
        "epochs_seen": 0,
        "steps_seen": 0,
        "completed": False,
    }

    if not log_text:
        return result

    for line in log_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        _extract_values(line, result)

    # Set final values (last in list)
    if result["loss"]:
        result["final_loss"] = result["loss"][-1]
    if result["accuracy"]:
        result["final_accuracy"] = result["accuracy"][-1]
    if result["throughput"]:
        result["final_throughput"] = result["throughput"][-1]
    if result["latency"]:
        result["final_latency"] = result["latency"][-1]

    # Detect completion
    result["completed"] = _detect_completion(log_text)

    return result


def _extract_values(line: str, result: Dict[str, Any]) -> None:
    """Extract metric values from a single log line."""
    for metric_name, patterns in _PATTERNS.items():
        if metric_name in ("epoch", "step"):
            continue
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                try:
                    value = float(match.group(1))
                    result[metric_name].append(value)
                    break  # One match per metric per line is enough
                except (ValueError, IndexError):
                    pass

    # Track epochs and steps
    for pattern in _PATTERNS["epoch"]:
        match = pattern.search(line)
        if match:
            try:
                result["epochs_seen"] = max(result["epochs_seen"], int(match.group(1)))
            except (ValueError, IndexError):
                pass

    for pattern in _PATTERNS["step"]:
        match = pattern.search(line)
        if match:
            try:
                result["steps_seen"] = max(result["steps_seen"], int(match.group(1)))
            except (ValueError, IndexError):
                pass


def _detect_completion(log_text: str) -> bool:
    """Detect if training/evaluation completed normally."""
    completion_indicators = [
        "training complete",
        "finished training",
        "evaluation complete",
        "done",
        "training finished",
        "completed successfully",
        "total time",
        "finished",
    ]
    log_lower = log_text.lower()
    # Check the last 20 lines for completion indicators
    last_lines = log_lower.split("\n")[-20:]
    last_chunk = "\n".join(last_lines)
    return any(indicator in last_chunk for indicator in completion_indicators)


def parse_log_file(log_path: str, model_name: str = "") -> Dict[str, Any]:
    """Parse a log file.

    Args:
        log_path: Path to the log file.
        model_name: Name of the model.

    Returns:
        Same as parse_log().
    """
    try:
        with open(log_path, "r") as f:
            log_text = f.read()
        return parse_log(log_text, model_name)
    except FileNotFoundError:
        logger.error("Log file not found: %s", log_path)
        return {
            "loss": [], "accuracy": [], "throughput": [], "latency": [],
            "final_loss": None, "final_accuracy": None,
            "final_throughput": None, "final_latency": None,
            "epochs_seen": 0, "steps_seen": 0, "completed": False,
            "error": f"Log file not found: {log_path}",
        }

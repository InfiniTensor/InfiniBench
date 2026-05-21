#!/usr/bin/env python3
"""Loss analysis utilities for stability testing.

Detects anomalies (NaN/Inf), spikes, and trends in training loss values.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_anomalies(losses: Dict[int, float]) -> List[Dict]:
    """Detect NaN and Inf values in loss series.

    Args:
        losses: Dict mapping iteration number to loss value.

    Returns:
        List of anomaly dicts with iteration, type, and value.
    """
    anomalies = []
    for iteration, loss in sorted(losses.items()):
        if math.isnan(loss):
            anomalies.append(
                {"iteration": iteration, "type": "NaN", "value": str(loss)}
            )
        elif math.isinf(loss):
            anomalies.append(
                {"iteration": iteration, "type": "Inf", "value": str(loss)}
            )
    return anomalies


def detect_spike(
    losses: Dict[int, float],
    threshold: float = 5.0,
    window: int = 10,
) -> List[Dict]:
    """Detect abnormal spikes in loss values.

    A spike is detected when a loss value exceeds `threshold` standard
    deviations from the rolling mean of the preceding `window` iterations.

    Args:
        losses: Dict mapping iteration number to loss value.
        threshold: Number of standard deviations for spike detection.
        window: Rolling window size for mean/std calculation.

    Returns:
        List of spike dicts with iteration, loss, z_score, and rolling_mean.
    """
    if len(losses) < window + 1:
        return []

    spikes = []
    sorted_iters = sorted(losses.keys())

    for i in range(window, len(sorted_iters)):
        current_iter = sorted_iters[i]
        current_loss = losses[current_iter]

        window_iters = sorted_iters[i - window : i]
        window_losses = [losses[it] for it in window_iters]

        mean = sum(window_losses) / len(window_losses)
        variance = sum((x - mean) ** 2 for x in window_losses) / len(window_losses)
        std = math.sqrt(variance) if variance > 0 else 1e-8

        z_score = abs(current_loss - mean) / std if std > 0 else 0.0

        if z_score > threshold:
            spikes.append(
                {
                    "iteration": current_iter,
                    "loss": round(current_loss, 6),
                    "z_score": round(z_score, 2),
                    "rolling_mean": round(mean, 6),
                }
            )

    return spikes


def check_trend(
    losses: Dict[int, float],
    min_points: int = 5,
) -> Dict:
    """Check if loss values show a decreasing trend.

    Uses simple linear regression on the loss values. Returns whether
    the slope is negative (loss is decreasing).

    Args:
        losses: Dict mapping iteration number to loss value.
        min_points: Minimum number of data points required for trend analysis.

    Returns:
        Dict with:
            - decreasing: bool — whether loss is decreasing
            - slope: float — linear regression slope
            - confidence: str — 'high', 'medium', or 'low'
    """
    if len(losses) < min_points:
        return {
            "decreasing": False,
            "slope": 0.0,
            "confidence": "insufficient_data",
            "message": f"Only {len(losses)} points, need at least {min_points}",
        }

    sorted_iters = sorted(losses.keys())
    n = len(sorted_iters)
    x = [float(i) for i in range(n)]
    y = [losses[it] for it in sorted_iters]

    # Simple linear regression
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)

    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-12:
        return {
            "decreasing": False,
            "slope": 0.0,
            "confidence": "zero_variance",
        }

    slope = (n * sum_xy - sum_x * sum_y) / denominator

    # R-squared for confidence
    mean_y = sum_y / n
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + (sum_y - slope * sum_x) / n)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if r_squared > 0.7:
        confidence = "high"
    elif r_squared > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "decreasing": slope < 0,
        "slope": round(slope, 6),
        "r_squared": round(r_squared, 4),
        "confidence": confidence,
    }


def analyze_loss_series(
    losses: Dict[int, float],
    spike_threshold: float = 5.0,
    spike_window: int = 10,
) -> Dict:
    """Comprehensive analysis of a loss series.

    Args:
        losses: Dict mapping iteration number to loss value.
        spike_threshold: Z-score threshold for spike detection.
        spike_window: Rolling window size.

    Returns:
        Analysis dict with anomalies, spikes, trend, and summary.
    """
    anomalies = detect_anomalies(losses)
    spikes = detect_spike(losses, spike_threshold, spike_window)
    trend = check_trend(losses)

    loss_values = list(losses.values())
    valid_values = [v for v in loss_values if math.isfinite(v)]

    summary = {
        "total_iterations": len(losses),
        "has_nan": any(math.isnan(v) for v in loss_values),
        "has_inf": any(math.isinf(v) for v in loss_values),
        "anomaly_count": len(anomalies),
        "spike_count": len(spikes),
        "trend_decreasing": trend["decreasing"],
        "min_loss": min(valid_values) if valid_values else None,
        "max_loss": max(valid_values) if valid_values else None,
    }

    return {
        "anomalies": anomalies,
        "spikes": spikes,
        "trend": trend,
        "summary": summary,
    }

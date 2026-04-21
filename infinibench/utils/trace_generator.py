#!/usr/bin/env python3
"""
Trace file generator with multiple patterns
"""

import csv
import random
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum


class TracePattern(Enum):
    """Trace generation patterns"""

    RANDOM = "random"
    POISSON = "poisson"  # Poisson distribution
    BURSTY = "bursty"  # Bursty traffic
    CONSTANT = "constant"  # Constant interval
    STEP = "step"  # Step-wise increase


def generate_trace(
    output_file: str,
    num_requests: int = 50,
    pattern: TracePattern = TracePattern.RANDOM,
    min_interval: float = 0,
    max_interval: float = 100,
    burst_probability: float = 0.2,
    **kwargs,
) -> None:
    """
    Generate a trace file

    Args:
        output_file: Output file path
        num_requests: Number of requests
        pattern: Generation pattern
        min_interval: Minimum interval (ms)
        max_interval: Maximum interval (ms)
        burst_probability: Burst probability
    """

    # Generate timestamps
    timestamps = _generate_timestamps(
        num_requests, pattern, min_interval, max_interval, burst_probability, **kwargs
    )

    # Generate token counts
    input_tokens_list, output_tokens_list = _generate_token_counts(num_requests)

    # save file
    if output_file.endswith(".csv"):
        _save_to_csv(output_file, timestamps, input_tokens_list, output_tokens_list)
    else:
        _save_to_json(output_file, timestamps, input_tokens_list, output_tokens_list)

    print(f"Created trace file: {output_file} with {num_requests} requests")


def _generate_timestamps(
    num_requests: int,
    pattern: TracePattern,
    min_interval: float,
    max_interval: float,
    burst_probability: float,
    **kwargs,
) -> List[float]:
    """Generate timestamp sequence"""
    timestamps = [0.0]

    if pattern == TracePattern.CONSTANT:
        interval = (min_interval + max_interval) / 2
        for i in range(1, num_requests):
            timestamps.append(timestamps[-1] + interval)

    elif pattern == TracePattern.POISSON:
        # Poisson distribution
        lambda_rate = kwargs.get("lambda_rate", 10)
        for i in range(1, num_requests):
            interval = np.random.exponential(1 / lambda_rate) * 1000  # convert to ms
            timestamps.append(timestamps[-1] + interval)

    elif pattern == TracePattern.BURSTY:
        # Bursty pattern
        for i in range(1, num_requests):
            if random.random() < burst_probability:
                interval = random.uniform(0, min_interval * 2)
            else:
                interval = random.uniform(min_interval, max_interval)
            timestamps.append(timestamps[-1] + interval)

    elif pattern == TracePattern.STEP:
        # Step-wise pattern
        step_size = kwargs.get("step_size", 20)
        for i in range(1, num_requests):
            if i % step_size == 0:
                interval = max_interval
            else:
                interval = min_interval
            timestamps.append(timestamps[-1] + interval)

    else:  # RANDOM
        for i in range(1, num_requests):
            interval = random.uniform(min_interval, max_interval)
            timestamps.append(timestamps[-1] + interval)

    return timestamps


def _generate_token_counts(num_requests: int) -> tuple:
    """Generate token counts"""
    common_inputs = [64, 128, 256, 512, 1024]
    common_outputs = [64, 128, 256, 512]

    input_tokens = []
    output_tokens = []

    for i in range(num_requests):
        # 80% use common sizes, 20% random
        if random.random() < 0.8:
            input_tokens.append(random.choice(common_inputs))
            output_tokens.append(random.choice(common_outputs))
        else:
            input_tokens.append(random.randint(32, 2048))
            output_tokens.append(random.randint(16, 1024))

    return input_tokens, output_tokens


def _save_to_csv(
    output_file: str,
    timestamps: List[float],
    input_tokens: List[int],
    output_tokens: List[int],
):
    """Save as CSV format"""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "request_id",
                "arrival_timestamp_ms",
                "input_token_num",
                "output_token_num",
            ]
        )

        for i, (ts, inp, out) in enumerate(
            zip(timestamps, input_tokens, output_tokens)
        ):
            request_id = f"req-{i:04d}"
            writer.writerow([request_id, round(ts, 2), inp, out])


def _save_to_json(
    output_file: str,
    timestamps: List[float],
    input_tokens: List[int],
    output_tokens: List[int],
):
    """Save as JSON format"""
    data = []
    for i, (ts, inp, out) in enumerate(zip(timestamps, input_tokens, output_tokens)):
        data.append(
            {
                "request_id": f"req-{i:04d}",
                "arrival_timestamp_ms": round(ts, 2),
                "input_token_num": inp,
                "output_token_num": out,
            }
        )

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

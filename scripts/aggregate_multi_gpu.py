#!/usr/bin/env python3
"""
aggregate_multi_gpu.py — Aggregate multi-GPU operator test results.

Scans gpu_0/, gpu_1/, ..., gpu_N/ subdirectories under the given
multi_gpu_<timestamp> directory, extracts per-GPU metrics from each
operator result JSON, computes aggregate totals, and prints a summary
table.  Saves aggregated_results.json into the multi_gpu directory.

Usage:
    python scripts/aggregate_multi_gpu.py <multi_gpu_output_dir>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def find_result_json(gpu_dir: Path) -> Optional[Path]:
    """Find the operator result JSON in a gpu_<N>/ directory.

    Looks in gpu_dir/operator/*_results.json (standard Executor output).
    """
    operator_dir = gpu_dir / "operator"
    if operator_dir.is_dir():
        results = sorted(operator_dir.glob("*_results.json"))
        if results:
            return results[-1]  # latest if multiple

    # Fallback: search recursively for any *_results.json
    results = sorted(gpu_dir.rglob("*_results.json"))
    if results:
        return results[-1]

    return None


def extract_metrics(result_path: Path) -> Optional[Dict[str, Any]]:
    """Extract per-GPU metrics from a result JSON file."""
    try:
        with open(result_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Failed to read {result_path}: {e}", file=sys.stderr)
        return None

    metrics = data.get("metrics", [])
    extracted = {
        "gpu_id": None,  # filled by caller
        "testcase": data.get("testcase", "unknown"),
        "result_code": data.get("result_code", -1),
        "latency_ms": 0.0,
        "tflops": 0.0,
        "bandwidth_gbs": 0.0,
        "accuracy": "N/A",
    }

    for m in metrics:
        name = m.get("name", "")
        value = m.get("value")

        if name == "operator.latency":
            extracted["latency_ms"] = float(value) if value is not None else 0.0
        elif name == "operator.flops":
            extracted["tflops"] = float(value) if value is not None else 0.0
        elif name == "operator.bandwidth":
            extracted["bandwidth_gbs"] = float(value) if value is not None else 0.0
        elif name == "operator.tensor_accuracy":
            extracted["accuracy"] = str(value) if value is not None else "N/A"

    return extracted


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(per_gpu: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from per-GPU results."""
    valid = [r for r in per_gpu if r["result_code"] == 0]

    if not valid:
        return {
            "total_tflops": 0.0,
            "avg_latency_ms": 0.0,
            "total_bandwidth_gbs": 0.0,
            "accuracy": "FAIL",
            "all_passed": False,
        }

    total_tflops = sum(r["tflops"] for r in valid)
    avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)
    total_bandwidth = sum(r["bandwidth_gbs"] for r in valid)
    all_passed = all(r["accuracy"] == "PASS" for r in valid)

    return {
        "total_tflops": round(total_tflops, 4),
        "avg_latency_ms": round(avg_latency, 6),
        "total_bandwidth_gbs": round(total_bandwidth, 4),
        "accuracy": "PASS" if all_passed else "FAIL",
        "all_passed": all_passed,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_summary(per_gpu: List[Dict[str, Any]], aggregate: Dict[str, Any]) -> None:
    """Print a formatted summary table."""
    gpu_count = len(per_gpu)

    # Header
    print()
    print("=" * 78)
    print(f"  Multi-GPU Operator Test Summary  ({gpu_count} GPUs)")
    print("=" * 78)

    # Column layout
    hdr = f"{'GPU':>5}  {'Latency(ms)':>12}  {'TFLOPS':>12}  {'BW(GB/s)':>12}  {'Accuracy':>10}"
    print(hdr)
    print("-" * 78)

    for r in per_gpu:
        gpu_label = f"GPU {r['gpu_id']}" if r["result_code"] == 0 else f"GPU {r['gpu_id']} (FAIL)"
        line = (
            f"{gpu_label:>5}  "
            f"{r['latency_ms']:>12.6f}  "
            f"{r['tflops']:>12.4f}  "
            f"{r['bandwidth_gbs']:>12.4f}  "
            f"{r['accuracy']:>10}"
        )
        print(line)

    print("-" * 78)

    agg_line = (
        f"{'TOTAL':>5}  "
        f"{aggregate['avg_latency_ms']:>12.6f}  "
        f"{aggregate['total_tflops']:>12.4f}  "
        f"{aggregate['total_bandwidth_gbs']:>12.4f}  "
        f"{aggregate['accuracy']:>10}"
    )
    print(agg_line)
    print("=" * 78)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-GPU operator test results"
    )
    parser.add_argument(
        "multi_gpu_dir",
        help="Path to the multi_gpu_<timestamp> directory containing gpu_0/, gpu_1/, ...",
    )
    args = parser.parse_args()

    base_dir = Path(args.multi_gpu_dir)
    if not base_dir.is_dir():
        print(f"[ERROR] Directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover gpu_* subdirectories
    gpu_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("gpu_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not gpu_dirs:
        print(f"[ERROR] No gpu_* subdirectories found in {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(gpu_dirs)} GPU result directories in {base_dir}")

    # Extract per-GPU metrics
    per_gpu: List[Dict[str, Any]] = []
    for gpu_dir in gpu_dirs:
        gpu_id = int(gpu_dir.name.split("_")[1])
        result_file = find_result_json(gpu_dir)

        if result_file is None:
            print(f"  [WARN] No result JSON found for GPU {gpu_id}", file=sys.stderr)
            per_gpu.append({
                "gpu_id": gpu_id,
                "testcase": "unknown",
                "result_code": -1,
                "latency_ms": 0.0,
                "tflops": 0.0,
                "bandwidth_gbs": 0.0,
                "accuracy": "MISSING",
            })
            continue

        metrics = extract_metrics(result_file)
        if metrics is None:
            per_gpu.append({
                "gpu_id": gpu_id,
                "testcase": "unknown",
                "result_code": -1,
                "latency_ms": 0.0,
                "tflops": 0.0,
                "bandwidth_gbs": 0.0,
                "accuracy": "ERROR",
            })
            continue

        metrics["gpu_id"] = gpu_id
        per_gpu.append(metrics)

    # Aggregate
    aggregate = aggregate_results(per_gpu)

    # Print summary table
    testcase = per_gpu[0].get("testcase", "unknown") if per_gpu else "unknown"
    print_summary(per_gpu, aggregate)

    # Save aggregated JSON
    output_file = base_dir / "aggregated_results.json"
    output_data = {
        "type": "multi_gpu_aggregation",
        "gpu_count": len(gpu_dirs),
        "testcase": testcase,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_gpu": per_gpu,
        "aggregate": aggregate,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Aggregated results saved to: {output_file}")

    if not aggregate["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

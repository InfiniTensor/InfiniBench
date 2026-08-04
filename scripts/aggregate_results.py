#!/usr/bin/env python3
"""
Test Results Aggregator for InfiniMetrics

Reads individual result JSON files from the output/ directory and produces a
structured summary with per-operator, per-scale, per-dtype breakdowns, including
metric values (latency, TFLOPS, bandwidth) and pass/fail statistics.

Usage:
    # Aggregate all results in default output directory
    python scripts/aggregate_results.py

    # Specify input and output paths
    python scripts/aggregate_results.py --input ./output --output ./summary.json

    # Print human-readable table to console
    python scripts/aggregate_results.py --print

    # Filter by operator / scale / dtype
    python scripts/aggregate_results.py --filter-operator matmul --filter-scale large
"""

import argparse
import json
import logging
import numbers
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_result_files(root: Path) -> List[Path]:
    """Recursively find all *_results.json files under root."""
    return sorted(root.rglob("*_results.json"))


def parse_run_id(run_id: str) -> Dict[str, str]:
    """
    Parse run_id like 'opbench.matmul.small.128x128.float16' into components.
    Falls back to empty strings if the format doesn't match.
    """
    parts = run_id.split(".")
    if len(parts) >= 5 and parts[0] == "opbench":
        return {
            "operator": parts[1],
            "scale": parts[2],
            "shape": parts[3],
            "dtype": parts[4],
        }
    return {"operator": "", "scale": "", "shape": "", "dtype": ""}


def load_result(path: Path) -> Optional[Dict[str, Any]]:
    """Load a single result JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metric values from a result dict."""
    metrics_out = {}
    for m in result.get("metrics", []):
        name = m.get("name", "")
        value = m.get("value")
        unit = m.get("unit", "")
        metrics_out[name] = {"value": value, "unit": unit}
    return metrics_out


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------


def aggregate(
    files: List[Path],
    filter_operator: Optional[str] = None,
    filter_scale: Optional[str] = None,
    filter_dtype: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate results from all files into a structured summary."""

    all_records: List[Dict[str, Any]] = []

    # Counters
    total = 0
    passed = 0
    failed = 0

    # Group-by containers
    by_operator: Dict[str, Dict] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "records": []}
    )
    by_scale: Dict[str, Dict] = defaultdict(lambda: {"passed": 0, "failed": 0})
    by_dtype: Dict[str, Dict] = defaultdict(lambda: {"passed": 0, "failed": 0})
    by_operator_dtype: Dict[str, Dict] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "records": []}
    )

    for f in files:
        result = load_result(f)
        if result is None:
            continue

        run_id = result.get("run_id", f.stem)
        info = parse_run_id(run_id)
        config = result.get("config", {})
        inputs = config.get("inputs") or [{}]
        operator = info["operator"] or config.get("operator", "")
        dtype = info["dtype"] or inputs[0].get("dtype", "")

        # Apply filters
        if filter_operator and operator != filter_operator:
            continue
        if filter_scale and info["scale"] != filter_scale:
            continue
        if filter_dtype and dtype != filter_dtype:
            continue

        total += 1
        rc = result.get("result_code", -1)
        is_pass = rc == 0
        if is_pass:
            passed += 1
        else:
            failed += 1

        metrics = extract_metrics(result)
        error_msg = result.get("error_msg")

        record = {
            "run_id": run_id,
            "testcase": result.get("testcase", ""),
            "operator": operator,
            "scale": info["scale"],
            "shape": info["shape"],
            "dtype": dtype,
            "result_code": rc,
            "passed": is_pass,
            "error_msg": error_msg,
            "time": result.get("time", ""),
            "duration_sec": result.get("duration"),
            "metrics": metrics,
            "source_file": str(f),
        }
        all_records.append(record)

        op = record["operator"]
        scale = record["scale"]
        dtype = record["dtype"]

        by_operator[op]["passed" if is_pass else "failed"] += 1
        by_operator[op]["records"].append(record)
        by_scale[scale]["passed" if is_pass else "failed"] += 1
        by_dtype[dtype]["passed" if is_pass else "failed"] += 1
        op_dt_key = f"{op}/{dtype}"
        by_operator_dtype[op_dt_key]["passed" if is_pass else "failed"] += 1
        by_operator_dtype[op_dt_key]["records"].append(record)

    # Build per-operator metric summaries (only for passed tests with values)
    operator_metrics_summary: Dict[str, Dict] = {}
    for op, data in by_operator.items():
        latency_values = []
        flops_values = []
        bw_values = []
        for rec in data["records"]:
            if not rec["passed"]:
                continue
            m = rec["metrics"]
            lat = m.get("operator.latency", {}).get("value")
            if isinstance(lat, numbers.Real) and not isinstance(lat, bool):
                latency_values.append(lat)
            fl = m.get("operator.flops", {}).get("value")
            if isinstance(fl, numbers.Real) and not isinstance(fl, bool):
                flops_values.append(fl)
            bw = m.get("operator.bandwidth", {}).get("value")
            if isinstance(bw, numbers.Real) and not isinstance(bw, bool):
                bw_values.append(bw)

        summary: Dict[str, Any] = {
            "passed": data["passed"],
            "failed": data["failed"],
            "total": data["passed"] + data["failed"],
        }
        if latency_values:
            summary["latency_ms"] = {
                "min": round(min(latency_values), 4),
                "max": round(max(latency_values), 4),
                "avg": round(sum(latency_values) / len(latency_values), 4),
                "count": len(latency_values),
            }
        if flops_values:
            summary["tflops"] = {
                "min": round(min(flops_values), 4),
                "max": round(max(flops_values), 4),
                "avg": round(sum(flops_values) / len(flops_values), 4),
                "count": len(flops_values),
            }
        if bw_values:
            summary["bandwidth_gbs"] = {
                "min": round(min(bw_values), 4),
                "max": round(max(bw_values), 4),
                "avg": round(sum(bw_values) / len(bw_values), 4),
                "count": len(bw_values),
            }
        operator_metrics_summary[op] = summary

    # Build the detailed table (one row per test case)
    detailed_table = []
    for rec in all_records:
        row = {
            "run_id": rec["run_id"],
            "operator": rec["operator"],
            "scale": rec["scale"],
            "shape": rec["shape"],
            "dtype": rec["dtype"],
            "passed": rec["passed"],
            "result_code": rec["result_code"],
            "error_msg": rec["error_msg"],
        }
        for metric_name, metric_val in rec["metrics"].items():
            short = metric_name.replace("operator.", "")
            row[short] = metric_val.get("value")
        detailed_table.append(row)

    return {
        "generated_at": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": f"{passed / total * 100:.1f}%" if total > 0 else "N/A",
        "by_operator": operator_metrics_summary,
        "by_scale": {
            k: {
                "passed": v["passed"],
                "failed": v["failed"],
                "total": v["passed"] + v["failed"],
            }
            for k, v in by_scale.items()
        },
        "by_dtype": {
            k: {
                "passed": v["passed"],
                "failed": v["failed"],
                "total": v["passed"] + v["failed"],
            }
            for k, v in by_dtype.items()
        },
        "by_operator_dtype": {
            k: {"passed": v["passed"], "failed": v["failed"]}
            for k, v in by_operator_dtype.items()
        },
        "failed_details": [
            {
                "run_id": r["run_id"],
                "operator": r["operator"],
                "scale": r["scale"],
                "shape": r["shape"],
                "dtype": r["dtype"],
                "error_msg": r["error_msg"],
            }
            for r in all_records
            if not r["passed"]
        ],
        "detailed_table": detailed_table,
    }


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------


def print_summary(agg: Dict[str, Any]) -> None:
    """Print a human-readable summary to console."""

    print("=" * 72)
    print("  InfiniMetrics Test Results Summary")
    print("=" * 72)
    print(f"  Total  : {agg['total']}")
    print(f"  Passed : {agg['passed']}")
    print(f"  Failed : {agg['failed']}")
    print(f"  Rate   : {agg['pass_rate']}")
    print("=" * 72)

    # By operator
    print("\n-- By Operator --")
    print(
        f"  {'Operator':<12} {'Total':>5} {'Passed':>6} {'Failed':>6} {'Latency(ms)':>14} {'TFLOPS':>10} {'BW(GB/s)':>10}"
    )
    print("  " + "-" * 68)
    for op, data in sorted(agg["by_operator"].items()):
        lat = data.get("latency_ms", {})
        lat_str = f"{lat['avg']:.4f}" if lat else "-"
        fl = data.get("tflops", {})
        fl_str = f"{fl['avg']:.4f}" if fl else "-"
        bw = data.get("bandwidth_gbs", {})
        bw_str = f"{bw['avg']:.4f}" if bw else "-"
        print(
            f"  {op:<12} {data['total']:>5} {data['passed']:>6} {data['failed']:>6} "
            f"{lat_str:>14} {fl_str:>10} {bw_str:>10}"
        )

    # By scale
    print("\n-- By Scale --")
    print(f"  {'Scale':<10} {'Total':>5} {'Passed':>6} {'Failed':>6}")
    print("  " + "-" * 30)
    for scale, data in sorted(agg["by_scale"].items()):
        print(
            f"  {scale:<10} {data['total']:>5} {data['passed']:>6} {data['failed']:>6}"
        )

    # By dtype
    print("\n-- By Dtype --")
    print(f"  {'Dtype':<10} {'Total':>5} {'Passed':>6} {'Failed':>6}")
    print("  " + "-" * 30)
    for dtype, data in sorted(agg["by_dtype"].items()):
        print(
            f"  {dtype:<10} {data['total']:>5} {data['passed']:>6} {data['failed']:>6}"
        )

    # Failed details
    if agg["failed_details"]:
        print(f"\n-- Failed Tests ({len(agg['failed_details'])}) --")
        print(f"  {'Run ID':<45} {'Error':<30}")
        print("  " + "-" * 75)
        for fd in agg["failed_details"]:
            rid = fd["run_id"]
            err = (fd["error_msg"] or "Unknown")[:30]
            print(f"  {rid:<45} {err:<30}")

    # Detailed table for passed tests with metrics
    passed_rows = [r for r in agg["detailed_table"] if r["passed"]]
    if passed_rows:
        print(f"\n-- Passed Tests Detail ({len(passed_rows)}) --")
        print(
            f"  {'Operator':<10} {'Scale':<8} {'Shape':<14} {'Dtype':<10} "
            f"{'Latency(ms)':>12} {'TFLOPS':>10} {'BW(GB/s)':>10} {'Accuracy':>10}"
        )
        print("  " + "-" * 90)
        for r in passed_rows:
            lat = r.get("latency")
            lat_str = f"{lat:.4f}" if lat is not None else "-"
            fl = r.get("flops")
            fl_str = f"{fl:.4f}" if fl is not None else "-"
            bw = r.get("bandwidth")
            bw_str = f"{bw:.4f}" if bw is not None else "-"
            acc = r.get("tensor_accuracy", "-")
            print(
                f"  {r['operator']:<10} {r['scale']:<8} {r['shape']:<14} {r['dtype']:<10} "
                f"{lat_str:>12} {fl_str:>10} {bw_str:>10} {str(acc):>10}"
            )

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate InfiniMetrics test results from output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/aggregate_results.py\n"
            "  python scripts/aggregate_results.py --input ./output --print\n"
            "  python scripts/aggregate_results.py --filter-operator matmul --print\n"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        default="./output",
        help="Input directory containing result JSON files (default: ./output)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./aggregated_results.json",
        help="Output JSON file path (default: ./aggregated_results.json)",
    )
    parser.add_argument(
        "--print",
        dest="print_summary",
        action="store_true",
        help="Print human-readable summary table to console",
    )
    parser.add_argument(
        "--filter-operator",
        default=None,
        help="Only include results for this operator",
    )
    parser.add_argument(
        "--filter-scale",
        default=None,
        help="Only include results for this scale (small/medium/large)",
    )
    parser.add_argument(
        "--filter-dtype",
        default=None,
        help="Only include results for this dtype (float16/float32/bfloat16)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    files = discover_result_files(input_dir)
    if not files:
        print(f"No result files found in {input_dir}")
        return 1

    print(f"Found {len(files)} result file(s) in {input_dir}")

    agg = aggregate(
        files,
        filter_operator=args.filter_operator,
        filter_scale=args.filter_scale,
        filter_dtype=args.filter_dtype,
    )

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"Aggregated results saved to {output_path}")

    # Print to console if requested
    if args.print_summary:
        print_summary(agg)

    return 0


if __name__ == "__main__":
    sys.exit(main())

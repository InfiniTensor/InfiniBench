#!/usr/bin/env python3
"""Generate a human-readable compatibility test report from JSON results."""

import json
import sys
from pathlib import Path


def generate_report(result_file: str) -> str:
    with open(result_file) as f:
        data = json.load(f)

    lines = []
    lines.append("=" * 80)
    lines.append("  CUDA 兼容性测试报告")
    lines.append("=" * 80)

    # Basic info
    lines.append(f"\n运行 ID:   {data.get('run_id', 'N/A')}")
    lines.append(f"时间:      {data.get('time', 'N/A')}")
    lines.append(f"平台:      {data.get('config', {}).get('platform', 'N/A')}")
    lines.append(f"result_code: {data.get('result_code', 'N/A')}")

    # Extract metrics
    metrics = {m["name"]: m["value"] for m in data.get("metrics", [])}
    details = metrics.get("compatibility.cuda_samples.details", [])

    total = metrics.get("compatibility.cuda_samples.total", 0)
    compile_passed = metrics.get("compatibility.cuda_samples.compile_passed", 0)
    compile_failed = metrics.get("compatibility.cuda_samples.compile_failed", 0)
    compile_rate = metrics.get("compatibility.cuda_samples.compile_pass_rate", 0)
    run_passed = metrics.get("compatibility.cuda_samples.run_passed", 0)
    run_failed = metrics.get("compatibility.cuda_samples.run_failed", 0)
    run_rate = metrics.get("compatibility.cuda_samples.run_pass_rate", 0)

    # Summary table
    lines.append("\n" + "-" * 80)
    lines.append("  汇总")
    lines.append("-" * 80)
    lines.append(f"  {'指标':<20} {'值':<10} {'通过率':<10}")
    lines.append(f"  {'----':<20} {'----':<10} {'----':<10}")
    lines.append(f"  {'样例总数':<20} {total:<10}")
    lines.append(f"  {'编译通过':<20} {compile_passed:<10} {compile_rate}%")
    lines.append(f"  {'编译失败':<20} {compile_failed:<10}")
    lines.append(f"  {'运行通过':<20} {run_passed:<10} {run_rate}%")
    lines.append(f"  {'运行失败':<20} {run_failed:<10}")
    lines.append(f"  {'跳过(编译未过)':<20} {compile_passed - run_passed - run_failed:<10}")

    # Categorize results
    all_pass = []
    compile_fail = []
    run_fail = []

    for d in details:
        name = d["name"]
        if d["compile_result"] == "fail":
            compile_fail.append(d)
        elif d["run_result"] == "pass":
            all_pass.append(name)
        elif d["run_result"] == "fail":
            run_fail.append(d)

    # Passed list
    lines.append("\n" + "-" * 80)
    lines.append(f"  编译+运行全部通过 ({len(all_pass)}/{total})")
    lines.append("-" * 80)
    if all_pass:
        # Print in columns
        col_width = 35
        cols = max(1, 76 // col_width)
        for i in range(0, len(all_pass), cols):
            row = all_pass[i : i + cols]
            lines.append("  " + "".join(f"{n:<{col_width}}" for n in row))
    else:
        lines.append("  (无)")

    # Compile failures
    lines.append("\n" + "-" * 80)
    lines.append(f"  编译失败 ({len(compile_fail)}/{total})")
    lines.append("-" * 80)
    for d in compile_fail:
        lines.append(f"  [FAIL] {d['name']}")
        error = d.get("error", "")
        # Extract key error line
        for err_line in error.split("\n"):
            err_line = err_line.strip()
            if err_line and ("error" in err_line.lower() or "fatal" in err_line.lower()):
                lines.append(f"         -> {err_line[:70]}")
                break

    # Run failures
    lines.append("\n" + "-" * 80)
    lines.append(f"  运行失败 ({len(run_fail)}/{total})")
    lines.append("-" * 80)
    for d in run_fail:
        lines.append(f"  [FAIL] {d['name']}")
        error = d.get("error", "")
        # Show first meaningful error line
        for err_line in error.split("\n"):
            err_line = err_line.strip()
            if err_line and len(err_line) > 5:
                lines.append(f"         -> {err_line[:70]}")
                break

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_compat_report.py <result.json>")
        print("       python generate_compat_report.py <result.json> --csv output.csv")
        sys.exit(1)

    result_file = sys.argv[1]
    csv_mode = "--csv" in sys.argv

    report = generate_report(result_file)
    print(report)

    if csv_mode:
        csv_idx = sys.argv.index("--csv")
        csv_file = sys.argv[csv_idx + 1] if csv_idx + 1 < len(sys.argv) else "compat_report.csv"

        with open(result_file) as f:
            data = json.load(f)

        metrics = {m["name"]: m["value"] for m in data.get("metrics", [])}
        details = metrics.get("compatibility.cuda_samples.details", [])

        with open(csv_file, "w") as f:
            f.write("sample_name,compile_result,run_result,error\n")
            for d in details:
                error = d.get("error", "").replace('"', '""').replace("\n", " ")
                f.write(f"{d['name']},{d['compile_result']},{d['run_result']},\"{error}\"\n")

        print(f"\nCSV saved to: {csv_file}")


if __name__ == "__main__":
    main()

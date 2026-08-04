#!/usr/bin/env python3
"""
Operator Test Input Generator for InfiniMetrics

Generates standardized test input data covering small, medium, and large tensor
scales with various shape and dtype combinations. Outputs InfiniMetrics-compatible
JSON configs and .npy data files ready for operator benchmarking.

Usage:
    # Generate inputs for all supported operators
    python scripts/generate_operator_inputs.py --output ./test_inputs --seed 42

    # Generate for specific operators and dtypes
    python scripts/generate_operator_inputs.py --operators matmul add --dtypes float16 float32

    # Generate only small-scale tests
    python scripts/generate_operator_inputs.py --scales small medium

    # Dry run (print plan without generating files)
    python scripts/generate_operator_inputs.py --dry-run
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shape & dtype definitions
# ---------------------------------------------------------------------------

# Shapes grouped by scale: (label, shapes_list)
# Each shape is a 2-element tuple for the "base shape" that operators interpret.
MATMUL_SHAPES = {
    "small": [
        (64, 64),
        (128, 128),
        (256, 256),
    ],
    "medium": [
        (512, 512),
        (768, 1024),
        (1024, 768),
    ],
    "large": [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
}

ELEMENTWISE_SHAPES = {
    "small": [
        (64, 64),
        (128, 256),
        (256, 512),
    ],
    "medium": [
        (512, 1024),
        (1024, 1024),
        (2048, 512),
    ],
    "large": [
        (2048, 2048),
        (4096, 4096),
        (8192, 1024),
    ],
}

SUPPORTED_DTYPES = ["float16", "float32", "bfloat16"]

DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2}

# ---------------------------------------------------------------------------
# Operator specs
# ---------------------------------------------------------------------------

OP_INPUT_NAMES = {
    "matmul": ["a", "b"],
    "mm": ["a", "b"],
    "add": ["a", "b"],
    "sub": ["a", "b"],
    "mul": ["a", "b"],
    "div": ["a", "b"],
    "linear": ["input", "weight", "bias"],
}

# Devices that use InfiniOps framework (ATen fallback) instead of InfiniCore
INFINIOPS_DEVICES = {"cambricon", "ascend"}


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def get_input_shapes(operator: str, base_shape: Tuple[int, ...]) -> List[List[int]]:
    """Derive input shapes from a base shape for a given operator."""
    if operator in ("matmul", "mm"):
        m, k = base_shape[0], base_shape[1]
        n = k  # square by default
        return [[m, k], [k, n]]
    elif operator == "linear":
        m, k = base_shape[0], base_shape[1]
        n = k
        return [[m, k], [k, n], [n]]
    else:
        # element-wise: both inputs share the same shape
        return [list(base_shape), list(base_shape)]


def get_output_shape(operator: str, input_shapes: List[List[int]]) -> List[int]:
    """Calculate output shape from input shapes."""
    if operator in ("matmul", "mm"):
        m = input_shapes[0][0]
        n = input_shapes[1][1]
        return [m, n]
    elif operator == "linear":
        m = input_shapes[0][0]
        n = input_shapes[1][0]
        return [m, n]
    else:
        return input_shapes[0].copy()


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

NUMPY_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "bfloat16": np.float32,  # numpy lacks bfloat16; store as float32
    "int8": np.int8,
    "int32": np.int32,
}


def generate_tensor(
    shape: List[int],
    dtype: str,
    rng: np.random.Generator,
    distribution: str = "uniform",
) -> np.ndarray:
    """Generate a random tensor with the given distribution."""
    np_dtype = NUMPY_DTYPE_MAP[dtype]
    if distribution == "uniform":
        data = rng.uniform(-1.0, 1.0, shape).astype(np_dtype)
    elif distribution == "normal":
        data = rng.normal(0.0, 1.0, shape).astype(np_dtype)
    else:
        data = rng.uniform(-1.0, 1.0, shape).astype(np_dtype)
    return data


# ---------------------------------------------------------------------------
# JSON config builder
# ---------------------------------------------------------------------------


def build_test_config(
    operator: str,
    device: str,
    inputs: List[Dict[str, Any]],
    output_shape: List[int],
    dtype: str,
    warmup: int = 10,
    measured: int = 100,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Dict[str, Any]:
    """Build an InfiniMetrics-compatible test input dict."""
    framework = "InfiniOps" if device.lower() in INFINIOPS_DEVICES else "InfiniCore"
    if device.lower() == "cambricon":
        atol = rtol = 0.01
    return {
        "run_id": f"opbench.{operator}._",
        "testcase": f"operator.{framework}.{operator.capitalize()}",
        "config": {
            "operator": operator,
            "device": device,
            "data_base_dir": "",
            "inputs": inputs,
            "outputs": [
                {
                    "name": "output",
                    "shape": output_shape,
                    "dtype": dtype,
                }
            ],
            "warmup_iterations": warmup,
            "measured_iterations": measured,
            "tolerance": {"atol": atol, "rtol": rtol},
        },
        "metrics": [
            {"name": "operator.latency"},
            {"name": "operator.tensor_accuracy"},
            {"name": "operator.flops"},
            {"name": "operator.bandwidth"},
        ],
    }


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    """Represents one test case to generate."""

    operator: str
    scale: str
    shape_label: str
    base_shape: Tuple[int, ...]
    dtype: str
    index: int  # unique within the whole batch


def enumerate_test_cases(
    operators: List[str],
    scales: List[str],
    dtypes: List[str],
) -> List[TestCase]:
    """Enumerate all test case combinations."""
    cases: List[TestCase] = []
    idx = 0
    for op in operators:
        shape_table = (
            MATMUL_SHAPES if op in ("matmul", "mm", "linear") else ELEMENTWISE_SHAPES
        )
        for scale in scales:
            shapes = shape_table.get(scale, [])
            for shape in shapes:
                for dtype in dtypes:
                    cases.append(
                        TestCase(
                            operator=op,
                            scale=scale,
                            shape_label=f"{shape[0]}x{shape[1]}",
                            base_shape=shape,
                            dtype=dtype,
                            index=idx,
                        )
                    )
                    idx += 1
    return cases


def format_size_mb(shape: List[int], dtype: str) -> float:
    """Calculate tensor size in MB."""
    elements = 1
    for d in shape:
        elements *= d
    return elements * DTYPE_BYTES.get(dtype, 4) / (1024 * 1024)


def generate_all(
    cases: List[TestCase],
    output_dir: Path,
    device: str,
    seed: int,
    warmup: int,
    measured: int,
) -> List[Path]:
    """Generate data files and JSON configs for all test cases."""
    rng = np.random.default_rng(seed)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Also produce a single combined JSON with all test cases
    all_configs: List[Dict[str, Any]] = []

    written: List[Path] = []
    total_data_mb = 0.0

    for tc in cases:
        input_shapes = get_input_shapes(tc.operator, tc.base_shape)
        output_shape = get_output_shape(tc.operator, input_shapes)
        input_names = OP_INPUT_NAMES.get(tc.operator, ["a", "b"])

        # Restrict input_names to actual number of inputs
        input_names = input_names[: len(input_shapes)]

        inputs_config: List[Dict[str, Any]] = []
        for name, shape in zip(input_names, input_shapes):
            # Generate data
            data = generate_tensor(shape, tc.dtype, rng)
            shape_str = "x".join(str(d) for d in shape)
            npy_name = f"{tc.operator}_{tc.scale}_{shape_str}_{tc.dtype}_{name}.npy"
            npy_path = data_dir / npy_name
            np.save(npy_path, data)

            total_data_mb += format_size_mb(shape, tc.dtype)

            inputs_config.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": tc.dtype,
                    "file_path": str(npy_path.resolve()),
                    "init_mode": "random",
                }
            )

        # Build run_id
        run_id = f"opbench.{tc.operator}.{tc.scale}.{tc.shape_label}.{tc.dtype}"

        config = build_test_config(
            operator=tc.operator,
            device=device,
            inputs=inputs_config,
            output_shape=output_shape,
            dtype=tc.dtype,
            warmup=warmup,
            measured=measured,
        )
        config["run_id"] = run_id
        config["config"]["data_base_dir"] = str(data_dir.resolve())

        # Write individual JSON
        json_name = f"{run_id}.json"
        json_path = config_dir / json_name
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        written.append(json_path)

        all_configs.append(config)

        logger.info(
            f"  [{tc.index + 1}/{len(cases)}] {run_id} "
            f"(data ~{format_size_mb(output_shape, tc.dtype):.2f} MB output)"
        )

    # Write combined JSON (list of all configs)
    combined_path = output_dir / "all_test_inputs.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_configs, f, indent=2, ensure_ascii=False)
    written.append(combined_path)

    logger.info(f"Total data generated: {total_data_mb:.2f} MB")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standardized operator test inputs for InfiniMetrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/generate_operator_inputs.py --output ./test_inputs --seed 42\n"
            "  python scripts/generate_operator_inputs.py --operators matmul --dtypes float16\n"
            "  python scripts/generate_operator_inputs.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./operator_test_inputs",
        help="Output directory (default: ./operator_test_inputs)",
    )
    parser.add_argument(
        "--operators",
        nargs="+",
        default=["matmul", "add", "sub", "mul", "div"],
        choices=["matmul", "mm", "add", "sub", "mul", "div", "linear"],
        help="Operators to generate inputs for (default: matmul add sub mul div)",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float16", "float32", "bfloat16"],
        choices=SUPPORTED_DTYPES,
        help="Data types (default: float16 float32 bfloat16)",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=["small", "medium", "large"],
        choices=["small", "medium", "large"],
        help="Tensor scale categories (default: small medium large)",
    )
    parser.add_argument(
        "--device",
        default="nvidia",
        help="Target device (default: nvidia)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--measured",
        type=positive_int,
        default=100,
        help="Measured iterations (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without generating files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    cases = enumerate_test_cases(args.operators, args.scales, args.dtypes)

    # Group by operator for summary
    from collections import Counter

    op_counts = Counter(c.operator for c in cases)
    scale_counts = Counter(c.scale for c in cases)
    dtype_counts = Counter(c.dtype for c in cases)

    print("=" * 60)
    print("Operator Test Input Generator")
    print("=" * 60)
    print(f"  Total test cases : {len(cases)}")
    print(f"  Operators        : {dict(op_counts)}")
    print(f"  Scales           : {dict(scale_counts)}")
    print(f"  Dtypes           : {dict(dtype_counts)}")
    print(f"  Device           : {args.device}")
    print(f"  Seed             : {args.seed}")
    print(f"  Output           : {args.output}")

    # Estimate total data size
    est_mb = 0.0
    for tc in cases:
        input_shapes = get_input_shapes(tc.operator, tc.base_shape)
        for s in input_shapes:
            est_mb += format_size_mb(s, tc.dtype)
    print(f"  Est. data size   : {est_mb:.1f} MB")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run - test case details:")
        print("-" * 60)
        print(f"{'#':<4} {'Operator':<10} {'Scale':<8} {'Shape':<14} {'Dtype':<10}")
        print("-" * 60)
        for tc in cases:
            input_shapes = get_input_shapes(tc.operator, tc.base_shape)
            shapes_str = " @ ".join("x".join(str(d) for d in s) for s in input_shapes)
            print(
                f"{tc.index:<4} {tc.operator:<10} {tc.scale:<8} "
                f"{shapes_str:<24} {tc.dtype:<10}"
            )
        print("-" * 60)
        print(f"Total: {len(cases)} test cases")
        return 0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {len(cases)} test cases...")
    written = generate_all(
        cases=cases,
        output_dir=output_dir,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        measured=args.measured,
    )

    # Save generation metadata
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "seed": args.seed,
        "device": args.device,
        "operators": args.operators,
        "scales": args.scales,
        "dtypes": args.dtypes,
        "warmup": args.warmup,
        "measured": args.measured,
        "total_cases": len(cases),
        "total_files": len(written),
        "estimated_data_mb": round(est_mb, 2),
        "cases": [
            {
                "operator": tc.operator,
                "scale": tc.scale,
                "base_shape": list(tc.base_shape),
                "dtype": tc.dtype,
            }
            for tc in cases
        ],
    }
    meta_path = output_dir / "_generation_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Generated {len(written)} files in {output_dir}/")
    print(f"  configs/    - individual JSON test configs")
    print(f"  data/       - .npy tensor data files")
    print(f"  all_test_inputs.json - combined config (use with main.py)")
    print(f"  _generation_metadata.json - generation info")
    print(f"\nUsage:")
    print(f"  python main.py {output_dir}/configs/")
    print(f"  python main.py {output_dir}/all_test_inputs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

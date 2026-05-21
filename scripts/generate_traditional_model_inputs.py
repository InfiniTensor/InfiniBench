#!/usr/bin/env python3
"""Generate test input JSON files for traditional model benchmarks.

Creates input JSONs for all model x platform combinations.

Usage:
    python scripts/generate_traditional_model_inputs.py \
        --output ./input_jsons \
        --platforms nvidia cambricon ascend
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infinimetrics.traditional_models.model_registry import (
    get_all_models,
    get_all_categories,
    get_platform_env_name,
)


def generate_inputs(
    output_dir: str,
    platforms: list = None,
    mode: str = "train",
) -> list:
    """Generate input JSON files for traditional model benchmarks.

    Args:
        output_dir: Directory to write JSON files.
        platforms: List of platform names. Defaults to all platforms.
        mode: "train" or "eval".

    Returns:
        List of generated file paths.
    """
    if platforms is None:
        platforms = [
            "nvidia", "metax", "iluvatar", "hygon", "moore",
            "cambricon", "ascend",
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    models = get_all_models()
    generated = []

    for platform in platforms:
        platform_inputs = []
        for model in models:
            testcase = f"traditional.{mode.capitalize()}.{model['category']}/{model['name']}"
            input_data = {
                "run_id": f"trad_{mode}_{platform}_{model['name']}",
                "testcase": testcase,
                "config": {
                    "model_category": model["category"],
                    "model_name": model["name"],
                    "mode": mode,
                    "platform": platform,
                    "timeout": 600,
                },
                "metrics": [],
            }
            platform_inputs.append(input_data)

        # Write per-platform file
        filename = f"traditional_{mode}_{platform}.json"
        filepath = output_path / filename
        with open(filepath, "w") as f:
            json.dump(platform_inputs, f, indent=2, ensure_ascii=False)
        generated.append(str(filepath))

        # Also write a single-file version (for batch dispatch)
        all_in_one = output_path / f"traditional_{mode}_{platform}_all.json"
        with open(all_in_one, "w") as f:
            json.dump(platform_inputs, f, indent=2, ensure_ascii=False)

    # Generate summary
    print(f"Generated {len(generated)} input files:")
    for path in generated:
        print(f"  {path}")
    print(f"Total models: {len(models)}")
    print(f"Total platforms: {len(platforms)}")
    print(f"Total test cases: {len(models) * len(platforms)}")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate test input JSONs for traditional model benchmarks"
    )
    parser.add_argument(
        "--output",
        default="./input_jsons/traditional",
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["nvidia", "metax", "iluvatar", "hygon", "moore", "cambricon", "ascend"],
        help="Target platforms",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Test mode",
    )

    args = parser.parse_args()
    generate_inputs(args.output, args.platforms, args.mode)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main entry point for InfiniBench test framework."""

import sys
import logging
import argparse
from typing import List, Dict, Any

from infinibench.dispatcher import Dispatcher
from infinibench.utils import load_inputs_from_paths

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_tests(
    inputs: List[Dict[str, Any]], default_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run tests using Dispatcher."""
    for test_input in inputs:
        test_input.setdefault("config", {}).setdefault(
            "output_dir", default_config.get("output_dir", "./output")
        )

    dispatcher = Dispatcher()
    return dispatcher.dispatch(inputs)


def print_summary(results: Dict[str, Any]) -> None:
    """Print test result summary."""
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    total = results["total_tests"]
    successful = results["successful_tests"]
    failed = results["failed_tests"]

    print(f"Total tests:   {total}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")

    if total > 0:
        print(f"Success rate:  {(successful/total)*100:.1f}%")

    if failed > 0:
        print("\nFailed tests:")
        for r in results["results"]:
            if r["result_code"] != 0:
                print(f"  - {r['testcase']}: code={r['result_code']}")

    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="InfiniBench - Test Orchestration Framework",
        epilog="Examples:\n  python main.py input.json\n  python main.py /path/to/inputs/\n  python main.py input.json --output ./results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories containing test specifications",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        inputs = load_inputs_from_paths(args.inputs)
        if not inputs:
            logger.error("No test inputs loaded")
            return 1

        default_config = {"output_dir": args.output}
        results = run_tests(inputs, default_config)
        print_summary(results)

        return 1 if results["failed_tests"] > 0 else 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())

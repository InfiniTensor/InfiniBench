#!/usr/bin/env python3
"""
Main entry point for inference evaluation
Usage:
cd ~/InfiniLM  # or vLLM directory
python /path/to/infinimetrics/inference/infer_main.py --config config.json
"""

import argparse
import os
import sys
import json
import logging
import traceback
from pathlib import Path

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('infer_benchmark.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool):
    """Set logging level"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)

def check_current_directory():
    """Check current directory and provide guidance"""
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    
    # Check if we're in InfiniLM or vLLM directory
    possible_frameworks = []
    
    if os.path.exists("scripts/jiuge.py"):
        possible_frameworks.append("InfiniLM")
    
    if os.path.exists("vllm") or "site-packages/vllm" in current_dir:
        possible_frameworks.append("vLLM")
    
    if possible_frameworks:
        logger.info(f"Detected framework(s): {', '.join(possible_frameworks)}")
    else:
        logger.warning("No known inference framework detected in current directory")
        logger.warning("Please run this script from either:")
        logger.warning("  1. InfiniLM directory (contains scripts/jiuge.py)")
        logger.warning("  2. vLLM directory or vLLM installation directory")

def load_config(config_file: str):
    """Load configuration file"""
    from infer_config import InferConfigManager
    
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
    
    try:
        config = InferConfigManager.load_config(config_file)
        logger.info(f"Configuration loaded successfully: {config.run_id}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

def check_dependencies():
    """Check for required dependencies"""
    from infer_runner_factory import InferRunnerFactory
    
    dependencies = InferRunnerFactory.check_dependencies()
    
    logger.info("Dependency check:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {dep}")
    
    # Check for essential dependencies
    if not dependencies["numpy"]:
        logger.warning("NumPy is not installed. Some statistics may not be available.")
    
    return dependencies

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Unified Inference Benchmark Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  1. Run from InfiniLM directory:
     cd ~/InfiniLM
     python /path/to/infinimetrics/inference/infer_main.py --config config.json
     
  2. Run from vLLM directory:
     cd ~/vllm
     python /path/to/infinimetrics/inference/infer_main.py --config config.json
     
  3. Enable verbose logging:
     python infer_main.py --config config.json --verbose
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Path to configuration file (JSON format)"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, do not execute tests"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from configuration"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    setup_logging(args.verbose)
    
    logger.info("=" * 60)
    logger.info("Unified Inference Benchmark Framework")
    logger.info("=" * 60)
    
    # Check current directory
    check_current_directory()
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        sys.exit(0)
    
    if not args.config:
        parser.error("the following arguments are required: --config")

    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified
    if args.output_dir:
        config.output_dir = args.output_dir
        logger.info(f"Output directory overridden: {config.output_dir}")
    
    # Validate configuration
    from infer_config import InferConfigManager
    errors = InferConfigManager.validate_config(config)
    
    if errors:
        logger.warning("Configuration validation warnings:")
        for error in errors:
            logger.warning(f"  - {error}")
    
    if args.validate_only:
        logger.info("Configuration validation completed")
        if not errors:
            logger.info("✓ Configuration is valid")
        else:
            logger.warning("⚠ Configuration has warnings but may still work")
        sys.exit(0)
    
    # Check dependencies
    dependencies = check_dependencies()
    
    # Check if framework is available
    if config.framework.value == "infinilm" and not dependencies["infinilm"]:
        logger.error("InfiniLM not detected in current directory")
        logger.error("Please run this script from InfiniLM directory")
        sys.exit(1)
    
    if config.framework.value == "vllm" and not dependencies["vllm"]:
        logger.error("vLLM not detected or not installed")
        logger.error("Please install vLLM or run from vLLM directory")
        sys.exit(1)
    
    try:
        # Create Runner and Adapter
        from infer_runner_factory import InferRunnerFactory
        runner, adapter = InferRunnerFactory.create_runner_and_adapter(config)
        
        # Run benchmark
        logger.info(f"Starting benchmark: {config.run_id}")
        logger.info(f"Mode: {config.mode.value}, Framework: {config.framework.value}")
        
        result_file = runner.run()
        
        if not isinstance(result_file, str):
            logger.error(f"Expected string result file path, got: {type(result_file)}")
            if isinstance(result_file, dict):
                # Emergency handling: print results directly
                logger.info("Results (dict format):")
                logger.info(json.dumps(result_file, indent=2))
                # Attempt to save to file
                emergency_file = Path(config.output_dir) / "infer" / f"emergency_{config.run_id}_results.json"
                with open(emergency_file, 'w') as f:
                    json.dump(result_file, f, indent=2)
                result_file = str(emergency_file)
            else:
                raise TypeError(f"Result file must be string, got {type(result_file)}")

        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {result_file}")
        
        # Output result location
        result_path = Path(result_file)
        if result_path.exists():
            with open(result_path, 'r') as f:
                result_data = json.load(f)
                success = result_data.get("success", 0)
                logger.info(f"Benchmark success status: {success}")
        else:
            logger.warning(f"Result file not found: {result_file}")

        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("This may be because:")
        logger.error("  1. You're not in the correct framework directory")
        logger.error("  2. The framework is not properly installed")
        logger.error("  3. The adapter implementation is missing")
        return 1
        
    except Exception as e:
        logger.error(f"Benchmark failed with error: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('framework_train.log') # optional: output to file
    ]
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import ConfigManager, create_gpu_monitor

def create_training_runner(framework, config_manager, gpu_monitor):
    """Factory function to create training runner"""
    logger.info(f"Creating training runner for framework: {framework}")

    if framework.lower() == "megatron":
        from frameworks import MegatronRunner
        return MegatronRunner(config_manager, gpu_monitor)
    elif framework.lower() == "infinitrain":
        try:
            from frameworks import InfinitainRunner
            return InfinitrainRunner(config_manager, gpu_monitor)
        except ImportError:
            print("Warning: Infinitrain runner not found, using Megatron as fallback")
            from frameworks import MegatronRunner
            return MegatronRunner(config_manager, gpu_monitor)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config.json")
    parser.add_argument("--gpu-platform", default="nvidia", choices=["nvidia", "other"],
                       help="GPU platform for monitoring")
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file '{args.config}' not found")
        return 1

    try:
        # Load config to get basic info
        with open(args.config, "r") as f:
            config_data = json.load(f)

        # Get framework from config
        framework = config_data.get("config", {}).get("framework", "megatron")
        run_id = config_data.get("config", {}).get("run_id", "unknown")
        testcase = config_data.get("config", {}).get("testcase", "unknown")
        
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Testcase: {testcase}")
        logger.info(f"Framework: {framework}")

        # Create configuration manager
        config_manager = ConfigManager(args.config)

        # Determine GPU platform: command line overrides config
        gpu_platform = args.gpu_platform
        if gpu_platform is None:
            gpu_platform = config_manager.gpu_platform
            logger.info(f"Using GPU platform from config: {gpu_platform}")
        else:
            logger.info(f"Using GPU platform from command line: {gpu_platform}")

        # Create GPU monitor
        gpu_monitor = create_gpu_monitor(gpu_platform)

        # Create training runner
        runner = create_training_runner(framework, config_manager, gpu_monitor)

        # Execute training
        result_json = runner.run()
        logger.info(f"Training completed. Results saved to: {result_json}")
        return 0

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON config file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())

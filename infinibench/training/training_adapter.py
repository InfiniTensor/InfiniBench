import logging
from typing import Dict, Any, Union, Optional
from datetime import datetime

from infinibench.adapter import BaseAdapter
from infinibench.input import TestInput
from infinibench.common.constants import (
    InfiniBenchJson,
    TestCategory,
    AcceleratorType,
)
from infinibench.common.testcase_utils import (
    generate_run_id_from_config,
    extract_testcase_components,
)
from infinibench.utils.accelerator_monitor import create_accelerator_monitor

logger = logging.getLogger(__name__)


class TrainingAdapter(BaseAdapter):
    """
    Training adapter for all training frameworks.

    Lifecycle (managed by Executor):
        1. setup(config) - Initialize resources
        2. process(test_input) - Execute training (monitoring inside)
        3. teardown() - Cleanup
    """

    def __init__(self):
        self.config = {}
        self.runner = None
        self.accelerator_monitor = None
        self._testcase = None

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Setup training resources.

        Args:
            config: Configuration dict from test_input (with injected _testcase, _run_id)
        """
        self.config = config
        self._testcase = config.get("_testcase", "training.unknown")

        test_dict = {
            "testcase": self._testcase,
            "run_id": config.get("_run_id"),
            "config": config,
        }
        self.run_id = generate_run_id_from_config(test_dict)

        # Extract framework from testcase or config
        testcase_info = extract_testcase_components(self._testcase)
        framework = testcase_info.get("framework", config.get("framework", "megatron"))

        logger.info(
            f"TrainingAdapter setup: testcase={self._testcase}, framework={framework}"
        )

        # Create accelerator monitor
        device_config = config.get("device", {})
        accelerator_type = device_config.get("gpu_platform", "nvidia")
        device_ids = device_config.get("device_ids")

        self.accelerator_monitor = create_accelerator_monitor(
            accelerator_type=accelerator_type,
            device_ids=device_ids,
            fallback_to_generic=True,
        )

        # Get resolved device count for runner
        resolved_device_count = self.accelerator_monitor.get_device_count()

        # Create framework-specific runner
        if framework == "megatron":
            from infinibench.training.frameworks.megatron_impl import MegatronImpl

            self.runner = MegatronImpl(config, resolved_device_count, self.run_id)
            logger.info(
                f"Created Megatron implementation with {resolved_device_count} devices, run_id={self.run_id}"
            )

        elif framework == "infinitrain":
            raise NotImplementedError("InfiniTrain implementation is not ready yet")

        else:
            raise ValueError(f"Unsupported training framework: {framework}")

    def process(self, test_input: Union[TestInput, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute training test.

        Monitoring lifecycle is contained within process():
            - start_monitoring() at beginning
            - stop_monitoring() in finally block

        Args:
            test_input: TestInput object or dict with testcase, config, etc.

        Returns:
            Standardized response dict with result_code, metrics, etc.
        """
        # Normalize input
        test_dict = self._normalize_test_input(test_input)
        if not test_dict:
            return self._create_error_response(
                "Invalid test input format", result_code=1
            )

        # Get or generate run_id
        run_id = generate_run_id_from_config(test_dict)
        testcase = test_dict.get("testcase", self._testcase)

        logger.info(f"Processing training test: {testcase} (run_id: {run_id})")

        # Start monitoring (per-test execution)
        if self.accelerator_monitor:
            self.accelerator_monitor.start_monitoring()
            logger.debug("Accelerator monitoring started")

        try:
            # Execute training
            result = self.runner.run()

            # Get peak memory after execution
            peak_memory_gb = (
                self.accelerator_monitor.get_peak_memory_gb()
                if self.accelerator_monitor
                else 0
            )

            # Build metrics list
            metrics = self._build_metrics(result, peak_memory_gb)

            # Build resolved info
            resolved = self._build_resolved_info()

            # Get framework from testcase
            testcase_info = extract_testcase_components(testcase)
            framework = testcase_info.get("framework", "megatron")

            # create an ordered configuration
            cleaned_config = self._clean_config(self.config)
            ordered_config = {}

            if hasattr(self.runner, "get_command"):
                command = self.runner.get_command()
                if command:
                    ordered_config["command"] = command

            for k, v in cleaned_config.items():
                if k != "command":
                    ordered_config[k] = v

            # Return standardized response
            response = {
                InfiniBenchJson.RUN_ID: run_id,
                InfiniBenchJson.TESTCASE: testcase,
                InfiniBenchJson.TIME: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                InfiniBenchJson.RESULT_CODE: 0,
                "success": 0,  # 0 = success (backward compatibility)
                InfiniBenchJson.CONFIG: cleaned_config,
                "resolved": resolved,
                InfiniBenchJson.METRICS: metrics,
            }

            logger.info(f"Training completed successfully: {testcase}")
            return response

        except Exception as e:
            logger.error(
                f"TrainingAdapter: Test failed for {testcase}\n"
                f"  Run ID: {run_id}\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            raise

        finally:
            # Always stop monitoring, even on failure
            if self.accelerator_monitor:
                self.accelerator_monitor.stop_monitoring()
                logger.debug("Accelerator monitoring stopped")

    def teardown(self) -> None:
        """Cleanup resources - do NOT stop monitoring here."""
        self.runner = None
        self.accelerator_monitor = None
        logger.info("TrainingAdapter teardown complete")

    def _build_metrics(self, result: Dict[str, Any], peak_memory_gb: float) -> list:
        """Build metrics list from runner result."""
        metrics = []

        # Throughput metrics
        if "throughput_csv" in result:
            metrics.append(
                {
                    "name": "train.throughput",
                    "type": "timeseries",
                    "raw_data_url": result["throughput_csv"],
                    "unit": "tokens/s/gpu",
                }
            )

        # Loss metrics
        if "loss_csv" in result:
            metrics.append(
                {
                    "name": "train.loss",
                    "type": "timeseries",
                    "raw_data_url": result["loss_csv"],
                    "unit": "",
                }
            )

        # PPL metrics
        if "ppl_csv" in result:
            metrics.append(
                {
                    "name": "train.ppl",
                    "type": "timeseries",
                    "raw_data_url": result["ppl_csv"],
                    "unit": None,
                }
            )

        # Peak memory (always include)
        metrics.append(
            {
                "name": "train.peak_memory_usage",
                "type": "scalar",
                "value": peak_memory_gb,
                "unit": "GB",
            }
        )

        return metrics

    def _build_resolved_info(self) -> Dict[str, Any]:
        """Build resolved hardware information using monitor's capabilities."""
        testcase_info = extract_testcase_components(self._testcase)
        framework_name = testcase_info.get(
            "framework", self.config.get("framework", "unknown")
        )

        if not self.accelerator_monitor:
            return {
                "nodes": 1,
                "gpus_per_node": 0,
                "device_used": 0,
                "framework": {
                    "name": framework_name,
                    "version": self._get_framework_version(framework_name),
                },
            }

        # Get device count
        device_count = self.accelerator_monitor.get_device_count()

        # TODO: In future, accelerator_monitor could provide a get_resolved_info() method
        # that returns platform-specific info (e.g., for Ascend, Cambricon)

        return {
            "nodes": 1,  # TODO: extract from multi-node config
            "gpus_per_node": device_count,
            "device_used": device_count,
            "accelerator_type": self.accelerator_monitor.accelerator_type.value,
            "framework": {
                "name": framework_name,
                "version": self._get_framework_version(framework_name),
            },
        }

    def _get_framework_version(self, framework: str) -> str:
        """Get framework version if possible."""
        if framework == "megatron":
            megatron_path = self.config.get("megatron_path", "")
            if megatron_path:
                pass
        elif framework == "infinitrain":
            pass
        return "unknown"

    def _clean_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal fields from config."""
        return {k: v for k, v in config.items() if not k.startswith("_")}

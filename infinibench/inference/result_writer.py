#!/usr/bin/env python3
"""Inference result writer"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from infinibench.utils.metrics import ScalarMetric, TimeseriesMetric
from infinibench.utils.path_utils import sanitize_filename

logger = logging.getLogger(__name__)


class InferenceResultWriter:
    """Writer for inference results"""

    def __init__(self, config, result_data):
        self.config = config
        self.result_data = result_data
        self.output_dir = Path(config.output_dir) / "infer"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_result(self) -> Dict[str, Any]:
        """Generate results in standard format"""
        # Save time series data
        self._save_timeseries_data()

        # Build standard JSON structure
        result = {
            "run_id": self.config.run_id,
            "testcase": self.config.testcase,
            "success": 0,  # 0 means success
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": self._build_config_dict(),
            "metrics": self._build_metrics_list(),
        }

        return result

    def _save_timeseries_data(self):
        """Save time series data to CSV files"""
        safe_run_id = sanitize_filename(self.config.run_id)

        # Save latency data
        if self.result_data.get("latency_data"):
            latency_file = self.output_dir / f"{safe_run_id}_infer_latency.csv"
            with open(latency_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "latency_ms"])
                for i, latency in enumerate(self.result_data["latency_data"]):
                    writer.writerow([i, latency])

        # Save TTFT data
        if self.result_data.get("ttft_data"):
            ttft_file = self.output_dir / f"{safe_run_id}_infer_ttft.csv"
            with open(ttft_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "ttft_ms"])
                for i, ttft in enumerate(self.result_data["ttft_data"]):
                    writer.writerow([i, ttft])

        # Save throughput data
        if self.result_data.get("throughput_data"):
            throughput_file = self.output_dir / f"{safe_run_id}_infer_throughput.csv"
            with open(throughput_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "throughput"])
                for i, throughput in enumerate(self.result_data["throughput_data"]):
                    writer.writerow([i, throughput])

    def _build_config_dict(self) -> Dict[str, Any]:
        """Build configuration dictionary"""
        config = {
            "framework": self.config.framework,
            "model": self.config.model,
            "model_path": self.config.model_path,
            "infer_args": self.config.infer_args,
            "warmup_iterations": self.config.warmup_iterations,
            "measured_iterations": self.config.measured_iterations,
        }

        # Add command string
        config["command"] = self._build_command_string()

        return config

    def _build_metrics_list(self) -> List[Dict[str, Any]]:
        """Build metrics list"""
        metrics = []
        safe_run_id = sanitize_filename(self.config.run_id)

        # Add required metrics based on mode
        if self.config.mode == "direct":
            # Required metrics for Direct mode
            required_metrics = [
                ("infer.ppl", "scalar", "placeholder", 0.0),
                ("infer.accuracy", "scalar", "placeholder", 0.0),
                (
                    "infer.peak_memory_usage",
                    "scalar",
                    "GB",
                    self.result_data.get("peak_memory_usage", 0.0),
                ),
                (
                    "infer.compute_latency",
                    "timeseries",
                    "ms",
                    f"./infer/{safe_run_id}_infer_latency.csv",
                ),
                (
                    "infer.ttft",
                    "timeseries",
                    "ms",
                    f"./infer/{safe_run_id}_infer_ttft.csv",
                ),
                (
                    "infer.direct_throughput_tps",
                    "timeseries",
                    "tokens/s/gpu",
                    f"./infer/{safe_run_id}_infer_throughput.csv",
                ),
            ]
        else:  # service mode
            # Required metrics for Service mode
            required_metrics = [
                ("infer.accuracy_mmlu", "scalar", None, None),
                (
                    "infer.peak_memory_usage",
                    "scalar",
                    "GB",
                    self.result_data.get("peak_memory_usage", 0.0),
                ),
                (
                    "infer.e2e_latency",
                    "timeseries",
                    "ms",
                    f"./infer/{safe_run_id}_infer_latency.csv",
                ),
                (
                    "infer.ttft",
                    "timeseries",
                    "ms",
                    f"./infer/{safe_run_id}_infer_ttft.csv",
                ),
                (
                    "infer.response_per_second",
                    "timeseries",
                    "req/s",
                    f"./infer/{safe_run_id}_infer_throughput.csv",
                ),
                (
                    "infer.compute_latency",
                    "timeseries",
                    "ms",
                    f"./infer/{safe_run_id}_infer_compute_latency.csv",
                ),
                (
                    "infer.max_throughput_tps",
                    "timeseries",
                    "tokens/s/gpu",
                    f"./infer/{safe_run_id}_infer_max_throughput.csv",
                ),
                (
                    "infer.success_rate",
                    "scalar",
                    "%",
                    self.result_data.get("success_rate", 1.0) * 100,
                ),
            ]

        # Create metric objects
        for name, metric_type, unit, value in required_metrics:
            if metric_type == "scalar":
                metrics.append(ScalarMetric(name, value, unit).to_dict())
            else:  # timeseries
                metrics.append(
                    {
                        "name": name,
                        "type": "timeseries",
                        "raw_data_url": value if value else None,
                        "unit": unit,
                    }
                )

        return metrics

    def _build_command_string(self) -> str:
        """Build command string"""
        if self.config.framework == "infinilm":
            if self.config.mode == "direct":
                return f"python scripts/jiuge.py --nvidia {self.config.model_path} 1"
            else:
                return f"python scripts/launch_server.py --model-path {self.config.model_path}"
        else:  # vllm
            if self.config.mode == "direct":
                return f"python -m vllm.benchmarks.benchmark_throughput --model {self.config.model_path}"
            else:
                return f"python -m vllm.entrypoints.api_server --model {self.config.model_path}"

    def _save_json_result(self, result: Dict[str, Any]):
        """Save JSON result file"""
        safe_run_id = sanitize_filename(self.config.run_id)
        json_file = self.output_dir / f"{safe_run_id}_results.json"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {json_file}")

#!/usr/bin/env python3
"""
Inference Runner Base Class
Defines common interfaces and template methods for all Runners
"""

import abc
import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re

from infer_config import InferConfig

logger = logging.getLogger(__name__)

class Metric:
    """Base metric class"""
    
    def __init__(self, name: str, metric_type: str, unit: Optional[str] = None):
        self.name = name
        self.type = metric_type
        self.unit = unit
        self.value = None
        self.raw_data_url = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "name": self.name,
            "type": self.type,
            "unit": self.unit
        }
        
        if self.type == "scalar":
            result["value"] = self.value
        elif self.type == "timeseries":
            result["raw_data_url"] = self.raw_data_url
        
        return result

class ScalarMetric(Metric):
    """Scalar metric"""
    
    def __init__(self, name: str, value: Any, unit: Optional[str] = None):
        super().__init__(name, "scalar", unit)
        self.value = value


class TimeseriesMetric(Metric):
    """Time series metric"""
    
    def __init__(self, name: str, raw_data_url: str, unit: Optional[str] = None):
        super().__init__(name, "timeseries", unit)
        self.raw_data_url = raw_data_url

class BenchmarkResult:
    """Benchmark result container"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Performance data
        self.latency_data: List[float] = []  # Latency data (milliseconds)
        self.ttft_data: List[float] = []     # Time to first token data (milliseconds)
        self.throughput_data: List[float] = []  # Throughput data
        
        # Additional data
        self.peak_memory_usage: Optional[float] = None  # Peak memory usage (GB)
        self.total_tokens: int = 0  # Total token count
        self.success_rate: float = 1.0  # Success rate
    
    def add_metric(self, metric: Metric):
        """Add a metric"""
        self.metrics.append(metric)
    
    def add_latency(self, latency_ms: float):
        """Add latency data"""
        self.latency_data.append(latency_ms)
    
    def add_ttft(self, ttft_ms: float):
        """Add TTFT data"""
        self.ttft_data.append(ttft_ms)
    
    def add_throughput(self, throughput: float):
        """Add throughput data"""
        self.throughput_data.append(throughput)

class InferRunnerBase(abc.ABC):
    """Inference Runner base class (Template Method Pattern)"""
    
    def __init__(self, config: InferConfig, adapter):
        self.config = config
        self.adapter = adapter
        self.result = BenchmarkResult()
        self.infer_dir: Optional[Path] = None
    
    def prepare_output_dir(self) -> Path:
        """Prepare output directory"""
        output_dir = Path(self.config.output_dir)
        self.infer_dir = output_dir / "infer"
        
        # Create directory
        self.infer_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory prepared: {self.infer_dir}")
        return self.infer_dir
    
    def save_timeseries_data(self):
        """Save time series data to CSV files"""
        if not self.infer_dir:
            raise ValueError("Output directory not prepared")
        
        # Save latency data
        if self.result.latency_data:
            # ✅ Clean filename
            safe_run_id = self._sanitize_filename(self.config.run_id)
            latency_file = self.infer_dir / f"{safe_run_id}_infer_latency.csv"
            
            with open(latency_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'latency_ms'])
                for i, latency in enumerate(self.result.latency_data):
                    writer.writerow([i, latency])
            
            # Add different metrics based on mode
            if self.config.mode.value == "direct":
                metric_name = "infer.compute_latency"
            else:
                metric_name = "infer.e2e_latency"
            
            self.result.add_metric(TimeseriesMetric(
                name=metric_name,
                raw_data_url=f"./infer/{latency_file.name}",
                unit="ms"
            ))
        
        # Save TTFT data
        if self.result.ttft_data:
            safe_run_id = self._sanitize_filename(self.config.run_id)
            ttft_file = self.infer_dir / f"{safe_run_id}_infer_ttft.csv"
            
            with open(ttft_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'ttft_ms'])
                for i, ttft in enumerate(self.result.ttft_data):
                    writer.writerow([i, ttft])
            
            self.result.add_metric(TimeseriesMetric(
                name="infer.ttft",
                raw_data_url=f"./infer/{ttft_file.name}",
                unit="ms"
            ))
        
        # Save throughput data
        if self.result.throughput_data:
            safe_run_id = self._sanitize_filename(self.config.run_id)
            throughput_file = self.infer_dir / f"{safe_run_id}_infer_throughput.csv"
            
            with open(throughput_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'throughput'])
                for i, throughput in enumerate(self.result.throughput_data):
                    writer.writerow([i, throughput])
            
            if self.config.mode.value == "direct":
                metric_name = "infer.direct_throughput_tps"
                unit = "tokens/s/gpu"
            else:
                metric_name = "infer.response_per_second"
                unit = "requests/s"
            
            self.result.add_metric(TimeseriesMetric(
                name=metric_name,
                raw_data_url=f"./infer/{throughput_file.name}",
                unit=unit
            ))
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate statistical information"""
        stats = {}
        
        try:
            import numpy as np
            
            # Calculate latency statistics
            if self.result.latency_data:
                stats['avg_latency'] = np.mean(self.result.latency_data)
                stats['p50_latency'] = np.percentile(self.result.latency_data, 50)
                stats['p95_latency'] = np.percentile(self.result.latency_data, 95)
                stats['p99_latency'] = np.percentile(self.result.latency_data, 99)
                stats['min_latency'] = np.min(self.result.latency_data)
                stats['max_latency'] = np.max(self.result.latency_data)
                stats['std_latency'] = np.std(self.result.latency_data)
            
            # Calculate TTFT statistics
            if self.result.ttft_data:
                stats['avg_ttft'] = np.mean(self.result.ttft_data)
                stats['p50_ttft'] = np.percentile(self.result.ttft_data, 50)
                stats['p95_ttft'] = np.percentile(self.result.ttft_data, 95)
                stats['p99_ttft'] = np.percentile(self.result.ttft_data, 99)
                stats['min_ttft'] = np.min(self.result.ttft_data)
                stats['max_ttft'] = np.max(self.result.ttft_data)
            
            # Calculate throughput statistics
            if self.result.throughput_data:
                stats['avg_throughput'] = np.mean(self.result.throughput_data)
                stats['max_throughput'] = np.max(self.result.throughput_data)
                stats['min_throughput'] = np.min(self.result.throughput_data)
            
            # Calculate total time and throughput rate
            if self.result.start_time and self.result.end_time:
                total_duration = self.result.end_time - self.result.start_time
                stats['total_duration'] = total_duration
                
                if self.config.mode.value == "direct":
                    if self.result.total_tokens > 0 and total_duration > 0:
                        stats['overall_throughput'] = self.result.total_tokens / total_duration
                
                elif self.config.mode.value == "service":
                    if len(self.result.latency_data) > 0 and total_duration > 0:
                        stats['requests_per_second'] = len(self.result.latency_data) / total_duration
            
            # Success rate
            stats['success_rate'] = self.result.success_rate
            
        except ImportError:
            logger.warning("NumPy not available, skipping statistics calculation")
        
        return stats

    def dump_json(self) -> str:
        """Output JSON results - ensure string path is returned"""
        if not self.infer_dir:
            raise ValueError("Output directory not prepared")

        safe_run_id = self._sanitize_filename(self.config.run_id)
        json_filename = f"{safe_run_id}_results.json"

        config_dict = {
            "run_id": self.config.run_id,
            "testcase": self.config.testcase,
            "success": 1 if self.result.success_rate >= 0.95 else 0,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "command": self._build_command_string(),
                "framework": self.config.framework.value,
                "model": self.config.model,
                "model_config": self.config.model_config,
                "train_dataset": self.config.train_dataset,
                "validation_dataset": self.config.validation_dataset,
                "test_dataset": self.config.test_dataset,
                "infer_args": self.config.infer_args.to_dict() if hasattr(self.config.infer_args, 'to_dict') else {},
                "warmup_iterations": self.config.warmup_iterations,
                "measured_iterations": self.config.measured_iterations
            },
            "metrics": []
        }

        # ✅ 1. First add all existing metrics
        for metric in self.result.metrics:
            config_dict["metrics"].append(metric.to_dict())

        # ✅ 2. Check and add missing metrics
        required_metrics = {
            "direct": [
                ("infer.peak_memory_usage", "scalar", "GB"),
                ("infer.compute_latency", "timeseries", "ms"),
                ("infer.ttft", "timeseries", "ms"),
                ("infer.direct_throughput_tps", "timeseries", "tokens/s/gpu")
            ],
            "service": [
                ("infer.accuracy_mmlu", "scalar", None),
                ("infer.peak_memory_usage", "scalar", "GB"),
                ("infer.e2e_latency", "timeseries", "ms"),
                ("infer.ttft", "timeseries", "ms"),
                ("infer.response_per_second", "timeseries", None),
                ("infer.compute_latency", "timeseries", "ms"),
                ("infer.max_throughput_tps", "timeseries", "tokens/s/gpu"),
                ("infer.success_rate", "scalar", "%")
            ]
        }
    
        mode = self.config.mode.value
        if mode in required_metrics:
            for metric_name, metric_type, unit in required_metrics[mode]:
                # Check if already exists
                if not any(m.get('name') == metric_name for m in config_dict['metrics']):
                    logger.debug(f"Adding missing metric: {metric_name}")
                
                    if metric_name == "infer.peak_memory_usage":
                        # Special handling: try to get peak memory
                        peak_memory = self.result.peak_memory_usage
                        if peak_memory is None:
                            try:
                                peak_memory = self.adapter.get_peak_memory_usage()
                            except Exception as e:
                                logger.warning(f"Failed to get peak memory usage: {e}")
                                peak_memory = 0.0
                    
                        # ✅ Ensure .to_dict() is called
                        config_dict["metrics"].append(ScalarMetric(
                            name=metric_name,
                            value=peak_memory,
                            unit=unit
                        ).to_dict())
                    
                    elif metric_name == "infer.success_rate":
                        # Special handling: success rate
                        success_rate = self.result.success_rate
                        config_dict["metrics"].append(ScalarMetric(
                            name=metric_name,
                            value=success_rate * 100,  # Convert to percentage
                            unit=unit
                        ).to_dict())
                    
                    elif metric_type == "scalar":
                        # Other scalar metrics (placeholder)
                        config_dict["metrics"].append(ScalarMetric(
                            name=metric_name,
                            value=0.0,  # Placeholder value
                            unit=unit
                        ).to_dict())  # ✅ Ensure .to_dict() is called
                    
                    elif metric_type == "timeseries":
                        # Time series metrics (create placeholder file)
                        file_created = self._ensure_metric_file(metric_name, safe_run_id)
                        if file_created:
                            raw_data_url = f"./infer/{safe_run_id}_{metric_name.replace('.', '_')}.csv"
                            config_dict["metrics"].append({
                                "name": metric_name,
                                "type": "timeseries",
                                "raw_data_url": raw_data_url,
                                "unit": unit
                            })

        # ✅ 3. Ensure no duplicate metrics
        seen_names = set()
        unique_metrics = []
        for metric in config_dict['metrics']:
            name = metric.get('name')
            if name and name not in seen_names:
                seen_names.add(name)
                unique_metrics.append(metric)
            elif name:
                logger.warning(f"Duplicate metric found: {name}")
    
        config_dict['metrics'] = unique_metrics

        # ✅ 4. Print debug information
        logger.info(f"Total metrics in JSON: {len(config_dict['metrics'])}")
        for metric in config_dict['metrics']:
            metric_name = metric.get('name', 'unknown')
            metric_type = metric.get('type', 'unknown')
            logger.debug(f"  - {metric_name} ({metric_type})")

        # ✅ 5. Save file
        json_file = self.infer_dir / json_filename
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {json_file}")
        return str(json_file)

    def _ensure_metric_file(self, metric_name: str, safe_run_id: str) -> bool:
        """Ensure metric file exists, create placeholder if not"""
        if not self.infer_dir:
            return False
    
        # Map metric name to filename
        file_mapping = {
            "infer.e2e_latency": f"{safe_run_id}_infer_latency.csv",
            "infer.compute_latency": f"{safe_run_id}_infer_compute_latency.csv",
            "infer.ttft": f"{safe_run_id}_infer_ttft.csv",
            "infer.response_per_second": f"{safe_run_id}_infer_throughput.csv",
            "infer.max_throughput_tps": f"{safe_run_id}_infer_max_throughput.csv",
            "infer.direct_throughput_tps": f"{safe_run_id}_infer_direct_throughput.csv"
        }
    
        if metric_name not in file_mapping:
            return False
    
        filename = file_mapping[metric_name]
        file_path = self.infer_dir / filename
    
        if not file_path.exists():
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'value'])
                    writer.writerow([0, 0.0])  # Placeholder data
                logger.debug(f"Created placeholder file: {filename}")
                return True
            except Exception as e:
                logger.warning(f"Failed to create placeholder file {filename}: {e}")
                return False
    
        return True

    def _create_max_throughput_placeholder(self):
        if not self.infer_dir:
            return
    
        safe_run_id = self._sanitize_filename(self.config.run_id)
        max_throughput_file = self.infer_dir / f"{safe_run_id}_infer_max_throughput.csv"
    
        with open(max_throughput_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'throughput'])
            writer.writerow([0, 0.0])  

    def _create_compute_latency_placeholder(self):
        if not self.infer_dir:
            return
    
        safe_run_id = self._sanitize_filename(self.config.run_id)
        compute_latency_file = self.infer_dir / f"{safe_run_id}_infer_compute_latency.csv"
    
        with open(compute_latency_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'latency_ms'])
            writer.writerow([0, 0.0])      

    def _build_command_string(self) -> str:
        """Build command string"""
        cmd_parts = []
        
        if self.config.framework.value == "infinilm":
            if self.config.mode.value == "direct":
                # Direct mode: use jiuge.py for direct inference
                cmd_parts.append("python scripts/jiuge.py --nvidia")
                cmd_parts.append(self.config.model_path)
                cmd_parts.append(str(self.config.infer_args.parallel.tp))
            
                # Add batch size parameter (if jiuge.py supports it)
                if self.config.mode.value == "direct":
                    if hasattr(self.config.infer_args, 'static_batch_size'):
                        cmd_parts.append(f"--batch-size {self.config.infer_args.static_batch_size}")
        
            else:  # service mode
                # Service mode: use launch_server.py to start inference service
                cmd_parts.append("python scripts/launch_server.py")
                cmd_parts.append(f"--model-path {self.config.model_path}")
                cmd_parts.append(f"--dev nvidia")
                cmd_parts.append(f"--ndev {self.config.infer_args.parallel.tp}")

                # Add common arguments
                if hasattr(self.config.infer_args, 'max_batch'):
                    cmd_parts.append(f"--max-batch {self.config.infer_args.max_batch}")
                elif hasattr(self.config.infer_args, 'max_seq_len'):
                    cmd_parts.append(f"--max-tokens {self.config.infer_args.max_seq_len}")

                # Add trace testing command (if trace is provided)
                if hasattr(self.config.infer_args, 'request_trace'):
                    trace_cmd = (
                        f"# Trace test: python trace_client.py "
                        f"--trace {self.config.infer_args.request_trace} "
                        f"--concurrency {self.config.infer_args.concurrency}"
                    )
                    cmd_parts.append(trace_cmd)

        else:  # vllm
            if self.config.mode.value == "direct":
                cmd_parts.append("python -m vllm.benchmarks.benchmark_throughput")
                cmd_parts.append(f"--model {self.config.model_path}")
                if hasattr(self.config.infer_args, 'static_batch_size'):
                    cmd_parts.append(f"--batch-size {self.config.infer_args.static_batch_size}")
            else:  # service
                cmd_parts.append("python -m vllm.entrypoints.api_server")
                cmd_parts.append(f"--model {self.config.model_path}")
                cmd_parts.append(f"--port 8000")
                cmd_parts.append(f"--tensor-parallel-size {self.config.infer_args.parallel.tp}")

                # Add trace serving command
                if hasattr(self.config.infer_args, 'request_trace'):
                    trace_cmd = (
                        f"# Trace test: python -m vllm.benchmarks.benchmark_serving "
                        f"--trace {self.config.infer_args.request_trace}"
                    )
                    cmd_parts.append(trace_cmd)
        
        return " ".join(cmd_parts)
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Clean filename by removing special characters
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        # Replace special characters with underscore
        sanitized = re.sub(r'[^\w\-_.]', '_', filename)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure reasonable filename length
        max_length = 255
        if len(sanitized) > max_length:
            # Keep first 100 chars and last 155 chars (add ... in between)
            prefix = sanitized[:100]
            suffix = sanitized[-155:] if len(sanitized) > 255 else ""
            sanitized = f"{prefix}...{suffix}"
        
        return sanitized
    
    def run(self) -> str:
        """Run template method - ensure string path is returned"""
        logger.info(f"Starting inference benchmark: {self.config.run_id}")
        logger.info(f"Testcase: {self.config.testcase}")
        logger.info(f"Framework: {self.config.framework.value}")
        logger.info(f"Mode: {self.config.mode.value}")
        
        try:
            # Prepare output directory
            self.prepare_output_dir()
            
            # Record start time
            self.result.start_time = time.time()
            
            # Template method: to be implemented by subclasses
            self.setup()
            self.execute()
            self.collect_metrics()
            
            # Record end time
            self.result.end_time = time.time()
            
            # Save data
            self.save_timeseries_data()
            
            # Output JSON
            result_file = self.dump_json()
            
            # ✅ Double check: ensure string is returned
            if not isinstance(result_file, str):
                logger.error(f"dump_json() returned {type(result_file)} instead of str")
                # Try conversion
                try:
                    result_file = str(result_file)
                except:
                    # If conversion fails, create default file
                    safe_run_id = self._sanitize_filename(self.config.run_id)
                    default_file = self.infer_dir / f"{safe_run_id}_emergency_results.json"
                    result_file = str(default_file)
                    logger.warning(f"Created emergency result file: {result_file}")
            
            # Print statistics
            self._print_statistics()
            
            logger.info(f"Benchmark completed successfully: {self.config.run_id}")
            
            # ✅ Final ensure string return
            return str(result_file)
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            raise
    
    def _print_statistics(self):
        """Print statistical information"""
        stats = self.calculate_statistics()
        
        logger.info("=" * 60)
        logger.info("BENCHMARK STATISTICS")
        logger.info("=" * 60)
        
        if 'avg_latency' in stats:
            logger.info(f"Average latency: {stats['avg_latency']:.2f} ms")
            if 'p95_latency' in stats:
                logger.info(f"P95 latency: {stats['p95_latency']:.2f} ms")
        
        if 'avg_ttft' in stats:
            logger.info(f"Average TTFT: {stats['avg_ttft']:.2f} ms")
            if 'p95_ttft' in stats:
                logger.info(f"P95 TTFT: {stats['p95_ttft']:.2f} ms")
        
        if 'avg_throughput' in stats:
            if self.config.mode.value == "direct":
                logger.info(f"Average throughput: {stats['avg_throughput']:.2f} tokens/s/gpu")
            else:
                logger.info(f"Average throughput: {stats['avg_throughput']:.2f} requests/s")
        
        if 'success_rate' in stats:
            logger.info(f"Success rate: {stats['success_rate']:.2%}")
        
        if 'total_duration' in stats:
            logger.info(f"Total duration: {stats['total_duration']:.2f} s")
        
        logger.info("=" * 60)
    
    @abc.abstractmethod
    def setup(self) -> None:
        """Set up execution environment (to be implemented by subclasses)"""
        pass
    
    @abc.abstractmethod
    def execute(self) -> None:
        """Execute inference test (to be implemented by subclasses)"""
        pass
    
    @abc.abstractmethod
    def collect_metrics(self) -> None:
        """Collect performance metrics (to be implemented by subclasses)"""
        pass

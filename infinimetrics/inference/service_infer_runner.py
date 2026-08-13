#!/usr/bin/env python3
"""
Service Inference Runner Implementation
Start service and run trace testing
"""

import asyncio
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from utils.gpu_monitor import create_gpu_monitor

from infer_runner_base import InferRunnerBase, TimeseriesMetric, ScalarMetric
from infer_config import InferConfig, ServiceInferArgs
from utils.trace_client import TraceClient, TraceClientConfig, RequestTrace
from utils.prompt_generator import create_prompt_generator

logger = logging.getLogger(__name__)

class ServiceInferRunner(InferRunnerBase):
    """Service Inference Runner"""
    
    def __init__(self, config: InferConfig, adapter):
        super().__init__(config, adapter)
        self.infer_args: ServiceInferArgs = config.infer_args
        
        # Trace related
        self.traces: List[RequestTrace] = []
        self.trace_stats: Dict[str, Any] = {}
        
        # Add GPU monitor
        self.gpu_monitor = None
        
        logger.info(f"ServiceInferRunner created for trace: {self.infer_args.request_trace}")
        logger.info(f"Concurrency: {self.infer_args.concurrency}")
        logger.info(f"Max sequence length: {self.infer_args.max_seq_len}")
    
    def setup(self) -> None:
        """Set up service inference environment"""
        logger.info("Setting up service inference environment")
        
        # 1. Create GPU monitor
        device_ids = self.config.device.device_ids
        if self.config.device.cpu_only:
            logger.info("CPU-only mode, GPU monitoring disabled")
            self.gpu_monitor = None
        else:
            self.gpu_monitor = create_gpu_monitor(
                gpu_platform=self.config.device.gpu_platform,
                device_ids=device_ids
            )

        # 2. Start GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.start_monitoring()
            logger.info(f"GPU monitoring started for devices: {device_ids}")
        
        # 3. Load trace file
        self._load_trace_data()
        
        # 4. Launch inference service
        logger.info(f"Launching inference service on port 8000")
        
        try:
            # Use adapter to launch service
            self.adapter.launch_service(port=8000)
            
            # Wait for service readiness
            max_wait_time = 120  # Maximum 120 seconds
            wait_interval = 3    # Check every 3 seconds
            
            logger.info("Waiting for service to be ready...")
            for i in range(max_wait_time // wait_interval):
                if self.adapter.is_service_ready(port=8000):
                    logger.info("Inference service is ready")
                    return
                
                logger.info(f"  Waiting... ({i * wait_interval}s elapsed)")
                time.sleep(wait_interval)
            
            raise TimeoutError("Inference service failed to start within timeout")
            
        except Exception as e:
            logger.error(f"Failed to setup service: {e}")
            
            # Stop GPU monitoring
            if self.gpu_monitor:
                self.gpu_monitor.stop_monitoring()

            # Ensure service is stopped
            try:
                self.adapter.stop_service()
            except:
                pass
            
            raise
    
    def execute(self) -> None:
        """Execute service inference test"""
        logger.info("Executing service inference benchmark")
        
        # Run asynchronous trace test
        asyncio.run(self._run_trace_async())
    
    async def _run_trace_async(self):
        """Asynchronously run trace test"""
        try:
            # 1. Create trace client configuration
            client_config = TraceClientConfig(
                api_url="http://localhost:8000",
                model_name=self.config.model,
                timeout_ms=self.infer_args.timeout_ms,
                warmup_requests=min(10, len(self.traces) // 10)  # 10% of requests for warmup
            )
            
            # 2. Create prompt generator
            # First get tokenizer (if adapter has loaded model)
            tokenizer = None
            if hasattr(self.adapter, 'tokenizer') and self.adapter.tokenizer:
                tokenizer = self.adapter.tokenizer
            
            prompt_generator = create_prompt_generator(
                tokenizer=tokenizer,
                method="random"  # Use random tokens
            )
            
            # 3. Load trace data
            self.traces = TraceClient.load_trace_file(
                self.infer_args.request_trace,
                prompt_generator
            )
            
            # 4. Use trace client to run test
            async with TraceClient(client_config) as client:
                # Run trace
                processed_traces, stats = await client.run_trace(
                    traces=self.traces,
                    concurrency=self.infer_args.concurrency,
                    warmup_requests=client_config.warmup_requests
                )
                
                # Save results
                self.traces = processed_traces
                self.trace_stats = stats
                
                # Save trace results to CSV
                if self.infer_dir:
                    client.save_results_to_csv(
                        processed_traces,
                        self.infer_dir,
                        self.config.run_id
                    )
        
        except Exception as e:
            logger.error(f"Trace test failed: {e}", exc_info=True)
            raise

        finally:
            # ✅ Stop GPU monitoring
            if self.gpu_monitor:
                self.gpu_monitor.stop_monitoring()
                peak_memory_gb = self.gpu_monitor.get_peak_memory_gb()
                logger.info(f"Peak GPU memory usage during test: {peak_memory_gb} GB")
                self.result.peak_memory_usage = peak_memory_gb

    def collect_metrics(self) -> None:
        """Collect service inference metrics"""
        logger.info("Collecting service inference metrics")
        
        # Extract data from trace statistics
        if self.trace_stats:
            # TTFT data
            ttfts = []
            for trace in self.traces:
                if trace.success and trace.ttft is not None:
                    ttfts.append(trace.ttft)
                    self.result.add_ttft(trace.ttft)
            
            # E2E latency data
            e2e_latencies = []
            for trace in self.traces:
                if trace.success and trace.e2e_latency is not None:
                    e2e_latencies.append(trace.e2e_latency)
                    self.result.add_latency(trace.e2e_latency)
            
            # Throughput data (requests/s)
            if self.trace_stats.get('total_duration', 0) > 0:
                rps = self.trace_stats.get('requests_per_second', 0)
                self.result.add_throughput(rps)
            
            # Success rate
            success_rate = self.trace_stats.get('success_rate', 0)
            self.result.success_rate = success_rate
            
            # Total tokens
            total_tokens = self.trace_stats.get('total_tokens', 0)
            self.result.total_tokens = total_tokens
            
            # Add scalar metrics
            if 'avg_ttft' in self.trace_stats:
                self.result.add_metric(ScalarMetric(
                    name="infer.avg_ttft",
                    value=self.trace_stats['avg_ttft'],
                    unit="ms"
                ))
            
            if 'avg_e2e_latency' in self.trace_stats:
                self.result.add_metric(ScalarMetric(
                    name="infer.avg_e2e_latency",
                    value=self.trace_stats['avg_e2e_latency'],
                    unit="ms"
                ))
            
            if 'throughput_tps' in self.trace_stats:
                self.result.add_metric(ScalarMetric(
                    name="infer.avg_throughput_tps",
                    value=self.trace_stats['throughput_tps'],
                    unit="tokens/s"
                ))
            
            # Success rate metric
            self.result.add_metric(ScalarMetric(
                name="infer.success_rate",
                value=success_rate * 100,  # Convert to percentage
                unit="%"
            ))
            
            # Total requests
            self.result.add_metric(ScalarMetric(
                name="infer.total_requests",
                value=self.trace_stats.get('total_requests', 0),
                unit="requests"
            ))
            
            # Record peak memory usage (if available)
            peak_memory = self.adapter.get_peak_memory_usage()
            if peak_memory:
                self.result.peak_memory_usage = peak_memory
                logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")
            
        else:
            logger.warning("No trace statistics available")
    
    def _load_trace_data(self):
        """Load trace data"""
        trace_file = self.infer_args.request_trace
        
        if not Path(trace_file).exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")
        
        logger.info(f"Loading trace data from: {trace_file}")
        
        # Create temporary prompt generator
        from utils.prompt_generator import create_prompt_generator
        temp_prompt_generator = create_prompt_generator(method="random")
        
        # Use trace client's method to load trace file
        self.traces = TraceClient.load_trace_file(
            trace_file, 
            temp_prompt_generator
        )
        
        logger.info(f"Loaded {len(self.traces)} requests from trace file")
        
        # Validate trace data
        self._validate_trace_data()
    
    def _validate_trace_data(self):
        """Validate trace data"""
        if not self.traces:
            raise ValueError("No trace data loaded")
        
        # Check if maximum tokens exceed model limits
        max_input_tokens = max(t.input_token_num for t in self.traces)
        max_output_tokens = max(t.output_token_num for t in self.traces)
        
        if max_input_tokens > self.infer_args.max_seq_len:
            logger.warning(f"Max input tokens ({max_input_tokens}) exceeds max_seq_len "
                          f"({self.infer_args.max_seq_len})")
        
        if max_output_tokens > self.infer_args.max_seq_len:
            logger.warning(f"Max output tokens ({max_output_tokens}) exceeds max_seq_len "
                          f"({self.infer_args.max_seq_len})")
        
        # Check timestamp order
        timestamps = [t.arrival_timestamp_ms for t in self.traces]
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            logger.warning("Trace timestamps are not sorted. Sorting now...")
            self.traces.sort(key=lambda x: x.arrival_timestamp_ms)
    
    def dump_json(self) -> str:
        """Override dump_json to generate standard format metrics - fixed version"""
        if not self.infer_dir:
            raise ValueError("Output directory not prepared")
        
        # Use base class method to generate basic JSON
        json_file = super().dump_json()
        
        # Read base class generated JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Build standard metrics array
        standard_metrics = []
        
        # 1. accuracy_mmlu (placeholder)
        standard_metrics.append({
            "name": "infer.accuracy_mmlu",
            "type": "scalar",
            "value": None,  # Placeholder, needs actual calculation
            "unit": None
        })
        
        # 2. e2e_latency (timeseries)
        latency_file = self.infer_dir / f"{self.config.run_id}_infer_latency.csv"
        if latency_file.exists():
            standard_metrics.append({
                "name": "infer.e2e_latency",
                "type": "timeseries",
                "raw_data_url": f"./infer/{latency_file.name}",
                "unit": "ms"
            })
        
        # 3. ttft (timeseries)
        ttft_file = self.infer_dir / f"{self.config.run_id}_infer_ttft.csv"
        if ttft_file.exists():
            standard_metrics.append({
                "name": "infer.ttft",
                "type": "timeseries",
                "raw_data_url": f"./infer/{ttft_file.name}",
                "unit": "ms"
            })
        
        # 4. peak_memory_usage (scalar) - Use GPU monitor to get real data
        # Get peak memory usage
        peak_memory = None
        
        if self.gpu_monitor:
            try:
                peak_memory = self.gpu_monitor.get_peak_memory_gb()
                logger.info(f"Real peak GPU memory usage: {peak_memory} GB")
            except Exception as e:
                logger.warning(f"Failed to get peak memory from GPU monitor: {e}")
                peak_memory = 0.0
        else:
            logger.warning("GPU monitor not available, using 0.0 GB")
        
        standard_metrics.append({
            "name": "infer.peak_memory_usage",
            "type": "scalar",
            "value": peak_memory,
            "unit": "GB"
        })
        
        # 5. response_per_second (timeseries)
        response_file = self.infer_dir / f"{self.config.run_id}_infer_throughput.csv"
        if response_file.exists():
            standard_metrics.append({
                "name": "infer.response_per_second",
                "type": "timeseries",
                "raw_data_url": f"./infer/{response_file.name}",
                "unit": None
            })
        else:
            # Add placeholder if no file
            standard_metrics.append({
                "name": "infer.response_per_second",
                "type": "timeseries",
                "raw_data_url": None,
                "unit": None
            })
        
        # 6. compute_latency (timeseries - placeholder)
        # Service mode may not have compute_latency file, add placeholder
        compute_latency_file = self.infer_dir / f"{self.config.run_id}_infer_compute_latency.csv"
        if compute_latency_file.exists():
            standard_metrics.append({
                "name": "infer.compute_latency",
                "type": "timeseries",
                "raw_data_url": f"./infer/{compute_latency_file.name}",
                "unit": "ms"
            })
        else:
            standard_metrics.append({
                "name": "infer.compute_latency",
                "type": "timeseries",
                "raw_data_url": None,
                "unit": "ms"
            })
        
        # 7. max_throughput_tps (timeseries)
        max_throughput_file = self.infer_dir / f"{self.config.run_id}_infer_max_throughput.csv"
        if max_throughput_file.exists():
            standard_metrics.append({
                "name": "infer.max_throughput_tps",
                "type": "timeseries",
                "raw_data_url": f"./infer/{max_throughput_file.name}",
                "unit": "tokens/s/gpu"
            })
        else:
            # Calculate max throughput from throughput data
            if hasattr(self.result, 'throughput_data') and self.result.throughput_data:
                max_throughput = max(self.result.throughput_data)
                standard_metrics.append({
                    "name": "infer.max_throughput_tps",
                    "type": "scalar",
                    "value": max_throughput,
                    "unit": "tokens/s/gpu"
                })
            else:
                standard_metrics.append({
                    "name": "infer.max_throughput_tps",
                    "type": "scalar",
                    "value": 0.0,
                    "unit": "tokens/s/gpu"
                })
        
        # Update metrics in data
        data["metrics"] = standard_metrics
        
        # Save back to file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Service metrics saved to: {json_file}")
        return str(json_file)
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up service inference resources")
        
        # Stop inference service
        try:
            self.adapter.stop_service()
        except Exception as e:
            logger.warning(f"Error stopping service: {e}")
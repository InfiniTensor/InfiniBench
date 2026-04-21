#!/usr/bin/env python3
"""Service inference engine"""

import asyncio
import logging
from .engine_utils import AcceleratorMonitorMixin
from .factory import create_service_manager

logger = logging.getLogger(__name__)


class ServiceInferenceEngine(AcceleratorMonitorMixin):
    """Service inference engine"""

    def _ia(self, key, default=None):
        ia = getattr(self.config, "infer_args", {})
        if isinstance(ia, dict):
            return ia.get(key, default)
        return getattr(ia, key, default)

    def __init__(self, config):
        self.config = config
        self.monitor = None
        self.service_manager = None
        self.result_data = {
            "latency_data": [],
            "ttft_data": [],
            "throughput_data": [],
            "peak_memory_usage": 0.0,
            "success_rate": 1.0,
        }

    def execute(self):
        """Execute service inference test"""
        logger.info(f"Starting service inference: {self.config.run_id}")

        # Run asynchronous trace test
        return asyncio.run(self._execute_async())

    async def _execute_async(self):
        """Execute trace test asynchronously"""
        try:
            self.service_manager = self._start_service()

            # Start accelerator monitoring
            self._start_accelerator_monitor()

            trace_results = await self._run_trace_test(self.service_manager)
            self._process_trace_results(trace_results)

            return self.result_data

        except Exception as e:
            logger.error(f"Service inference failed: {e}", exc_info=True)
            raise

        finally:
            try:
                self._stop_and_collect_monitor()
            except Exception as e:
                logger.warning(f"Failed to collect monitor: {e}")

            try:
                if self.service_manager:
                    self.service_manager.stop_service()
            except Exception as e:
                logger.warning(f"Failed to stop service: {e}")

    def _start_service(self):
        """Start inference service"""
        manager = create_service_manager(self.config)

        manager.start_service()
        return manager

    async def _run_trace_test(self, service_manager):
        """Run trace test"""
        from infinibench.utils.trace_client import TraceClient, TraceClientConfig

        # Create trace client
        client_config = TraceClientConfig(
            api_url=service_manager.get_service_url(),
            model_name=self.config.model,
            timeout_ms=self._ia("timeout_ms", 30000),
        )

        async with TraceClient(client_config) as client:
            # Load trace file
            traces = client.load_trace_file(
                self._ia("request_trace", ""), self._create_prompt_generator()
            )

            # Run trace
            processed_traces, stats = await client.run_trace(
                traces, concurrency=self._ia("concurrency", 32)
            )

            return processed_traces, stats

    def _process_trace_results(self, trace_results):
        processed_traces, stats = trace_results

        # TTFT / E2E
        for trace in processed_traces:
            ttft = getattr(trace, "ttft", None)
            e2e = getattr(trace, "e2e_latency", None)

            if ttft is not None:
                self.result_data["ttft_data"].append(float(ttft))
            if e2e is not None:
                self.result_data["latency_data"].append(float(e2e))

        #  success_rate
        if isinstance(stats, dict) and "success_rate" in stats:
            self.result_data["success_rate"] = stats["success_rate"]

        # throughput
        overall_rps = None

        if isinstance(stats, dict):
            req_cnt = stats.get("total_requests") or stats.get("request_count")
            wall_s = stats.get("wall_time_s") or stats.get("elapsed_time_s")

            if req_cnt and wall_s and wall_s > 0:
                overall_rps = float(req_cnt) / float(wall_s)

            # Compatibility with existing fields
            if overall_rps is None and stats.get("requests_per_second") is not None:
                overall_rps = float(stats["requests_per_second"])

        # fallback calculation
        if overall_rps is None:
            n = len(self.result_data["latency_data"])
            if n > 0:
                max_ms = max(self.result_data["latency_data"])
                if max_ms > 0:
                    overall_rps = n / (max_ms / 1000.0)

        if overall_rps is None:
            overall_rps = 0.0

        self.result_data["throughput_data"] = [overall_rps]

    def _create_prompt_generator(self):
        """Create prompt generator - returns a callable"""
        from infinibench.utils.prompt_generator import PromptGenerator

        # Create PromptGenerator instance
        pg_instance = PromptGenerator(method="random")

        # Return a wrapper function to make it callable
        def generate_prompt(token_num: int) -> str:
            """Wrapper function that calls pg_instance's generate method"""
            # Try different generation methods
            if hasattr(pg_instance, "generate_prompt"):
                return pg_instance.generate_prompt(token_num)
            elif hasattr(pg_instance, "generate"):
                return pg_instance.generate(token_num)
            else:
                # Fallback to simple random generation
                import random
                import string

                chars_per_token = 4
                total_chars = token_num * chars_per_token
                chars = string.ascii_letters + string.digits + " .,!?;:\n"
                return "".join(random.choices(chars, k=total_chars))

        return generate_prompt

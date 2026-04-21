#!/usr/bin/env python3
"""Direct inference engine"""

import logging
import time
from typing import List
from pathlib import Path
from .engine_utils import AcceleratorMonitorMixin
from .factory import create_framework_impl
from infinibench.utils.accelerator_monitor import create_accelerator_monitor

logger = logging.getLogger(__name__)


class DirectInferenceEngine(AcceleratorMonitorMixin):
    """Direct inference engine"""

    def __init__(self, config):
        self.config = config
        self.adapter = None
        self.monitor = None
        self.result_data = {
            "latency_data": [],
            "ttft_data": [],
            "throughput_data": [],
            "peak_memory_usage": 0.0,
            "total_tokens": 0,
            "success_rate": 1.0,
        }

    def execute(self):
        logger.info(f"Starting direct inference: {self.config.run_id}")
        try:
            # load model
            self.adapter = self._load_framework_adapter()

            # start accelerator monitor
            self._start_accelerator_monitor()

            # prompts
            prompts = self._generate_prompts()

            # warmup
            self._warmup(prompts)

            # measure
            self._run_measurements(prompts)

            # stop monitor & collect
            self._stop_and_collect_monitor()

            self._collect_total_tokens()
            return self.result_data

        except Exception as e:
            logger.error(f"Direct inference failed: {e}", exc_info=True)
            raise
        finally:
            self._unload_model()

    def _load_framework_adapter(self):
        """Load the frame adapter"""
        adapter = create_framework_impl(self.config)

        # load model
        logger.info("Loading model...")
        adapter.load_model()
        logger.info("Model loaded successfully")
        return adapter

    def _generate_prompts(self) -> List[str]:
        """Generate test prompts"""
        from infinibench.utils.prompt_generator import PromptGenerator

        ia = self.config.infer_args
        batch_size = (
            ia.get("static_batch_size", 1)
            if isinstance(ia, dict)
            else getattr(ia, "static_batch_size", 1)
        )
        prompt_tokens = (
            ia.get("prompt_token_num", 128)
            if isinstance(ia, dict)
            else getattr(ia, "prompt_token_num", 128)
        )

        total_needed = (
            self.config.warmup_iterations + self.config.measured_iterations
        ) * batch_size
        tokenizer = getattr(self.adapter, "tokenizer", None)

        generator = PromptGenerator(method="template", tokenizer=tokenizer)
        prompts = generator.generate_prompts(total_needed, prompt_tokens)
        logger.info(f"Generated {len(prompts)} prompts")
        return prompts

    def _warmup(self, prompts: List[str]):
        logger.info(f"Warmup: {self.config.warmup_iterations} iterations")

        ia = self.config.infer_args
        if isinstance(ia, dict):
            batch_size = ia.get("static_batch_size", 1)
            output_tokens = ia.get("output_token_num", 128)
            temperature = ia.get("temperature", 0.7)
            top_p = ia.get("top_p", 0.9)
            top_k = ia.get("top_k", 50)
        else:
            batch_size = getattr(ia, "static_batch_size", 1)
            output_tokens = getattr(ia, "output_token_num", 128)
            temperature = getattr(ia, "temperature", 0.7)
            top_p = getattr(ia, "top_p", 0.9)
            top_k = getattr(ia, "top_k", 50)

        for i in range(self.config.warmup_iterations):
            s = i * batch_size
            e = s + batch_size
            batch_prompts = prompts[s:e]
            if not batch_prompts:
                logger.warning(f"No prompts for warmup iter {i}")
                continue

            self.adapter.generate(
                batch_prompts,
                output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

    def _run_measurements(self, prompts: List[str]):
        """Perform measurement phase"""
        logger.info(f"Measurement: {self.config.measured_iterations} iterations")

        ia = self.config.infer_args
        if isinstance(ia, dict):
            batch_size = ia.get("static_batch_size", 1)
            output_tokens = ia.get("output_token_num", 128)
            temperature = ia.get("temperature", 0.7)
            top_p = ia.get("top_p", 0.9)
            top_k = ia.get("top_k", 50)
        else:
            batch_size = getattr(ia, "static_batch_size", 1)
            output_tokens = getattr(ia, "output_token_num", 128)
            temperature = getattr(ia, "temperature", 0.7)
            top_p = getattr(ia, "top_p", 0.9)
            top_k = getattr(ia, "top_k", 50)

        for i in range(self.config.measured_iterations):
            s = (self.config.warmup_iterations + i) * batch_size
            e = s + batch_size
            batch_prompts = prompts[s:e]
            if not batch_prompts:
                logger.warning(f"No prompts for measurement iter {i}")
                continue

            start = time.perf_counter()
            texts, latencies, ttfts = self.adapter.generate(
                batch_prompts,
                output_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            _ = (time.perf_counter() - start) * 1000

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                self.result_data["latency_data"].append(avg_latency)

                total_tokens = len(batch_prompts) * output_tokens
                throughput = (
                    (total_tokens * 1000) / avg_latency if avg_latency > 0 else 0
                )
                self.result_data["throughput_data"].append(throughput)

            if ttfts:
                avg_ttft = sum(ttfts) / len(ttfts)
                self.result_data["ttft_data"].append(avg_ttft)

    def _collect_total_tokens(self):
        ia = self.config.infer_args
        batch_size = (
            ia.get("static_batch_size", 1)
            if isinstance(ia, dict)
            else getattr(ia, "static_batch_size", 1)
        )
        output_tokens = (
            ia.get("output_token_num", 128)
            if isinstance(ia, dict)
            else getattr(ia, "output_token_num", 128)
        )
        self.result_data["total_tokens"] = (
            self.config.measured_iterations * batch_size * output_tokens
        )

    def _collect_peak_memory(self):
        if self.result_data.get("peak_memory_usage", 0) > 0:
            return

        if self.adapter and hasattr(self.adapter, "get_peak_memory_usage"):
            try:
                peak = self.adapter.get_peak_memory_usage()
                if peak:
                    self.result_data["peak_memory_usage"] = peak
            except Exception:
                pass

    def _unload_model(self):
        if self.adapter and hasattr(self.adapter, "unload_model"):
            try:
                self.adapter.unload_model()
                logger.info("Model unloaded")
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

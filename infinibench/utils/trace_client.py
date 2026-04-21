#!/usr/bin/env python3
"""
Trace client implementation
Reads requests from a trace file and sends them to the inference service
according to the specified time intervals.
"""

import asyncio
import aiohttp
import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from infinibench.common.constants import (
    DEFAULT_TIMEOUT_MS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

logger = logging.getLogger(__name__)


@dataclass
class RequestTrace:
    """Data of a single request in the trace file"""

    request_id: str
    arrival_timestamp_ms: float  # Timestamp in milliseconds
    input_token_num: int
    output_token_num: int

    # Runtime-calculated fields
    actual_prompt: Optional[str] = None
    start_time: Optional[float] = None
    ttft: Optional[float] = None  # Time To First Token
    e2e_latency: Optional[float] = None  # End-to-end latency
    total_tokens: int = 0
    success: bool = False
    error: Optional[str] = None


@dataclass
class TraceClientConfig:
    """Trace client configuration"""

    api_url: str
    model_name: str
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    max_retries: int = 3
    warmup_requests: int = 10
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P


class TraceClient:
    """Trace client"""

    def __init__(self, config: TraceClientConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_counter = 0
        self.semaphore: Optional[asyncio.Semaphore] = None

        # Performance statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000 + 10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _handle_request_error(
        self, trace: RequestTrace, error_type: str, error_msg: str
    ):
        """Unified request error handling"""
        trace.error = error_msg
        trace.success = False
        self.failed_requests += 1

        if error_type == "timeout":
            logger.error(f"Request {trace.request_id} timeout: {error_msg}")
        else:
            logger.error(f"Request {trace.request_id} failed: {error_msg}")

    async def _process_stream_response(
        self, response, trace: RequestTrace
    ) -> Tuple[Optional[float], int]:
        """Process streaming response and compute TTFT and token count"""
        first_token_received = False
        first_token_time = None
        ttft = None
        total_tokens = 0

        try:
            async for line in response.content:
                if not line:
                    continue

                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                # Parse and process data
                token_increment = self._parse_sse_data(
                    data_str, trace, first_token_received, first_token_time
                )

                if token_increment is not None:
                    # Record TTFT if this is the first token
                    if not first_token_received:
                        first_token_time = time.perf_counter()
                        ttft = (first_token_time - trace.start_time) * 1000
                        first_token_received = True
                        logger.debug(
                            f"First token received for {trace.request_id}: TTFT={ttft:.2f}ms"
                        )

                    total_tokens += token_increment

        except Exception as e:
            logger.warning(f"Stream processing error for {trace.request_id}: {e}")

        return ttft, total_tokens

    def _parse_sse_data(
        self,
        data_str: str,
        trace: RequestTrace,
        first_token_received: bool,
        first_token_time: Optional[float],
    ) -> Optional[int]:
        """Parse a single SSE data line"""
        try:
            data = json.loads(data_str)
            if "choices" not in data or len(data["choices"]) == 0:
                return None

            choice = data["choices"][0]

            # Check whether there is content
            if "delta" in choice and "content" in choice["delta"]:
                content = choice["delta"]["content"]

                # If non-empty content exists, count it as one token
                if content.strip():
                    return 1

            # Check whether generation is finished
            if choice.get("finish_reason"):
                return 0  # End marker

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in SSE stream: {data_str}")

        return None

    @staticmethod
    def load_trace_file(trace_file: str, prompt_generator) -> List[RequestTrace]:
        """Load a trace file"""
        traces = []

        # Verify document format
        if not trace_file.endswith(".csv"):
            logger.warning(
                f"Trace file {trace_file} does not have .csv extension. "
                f"Assuming CSV format anyway."
            )

        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Support different column name variants
                    request_id = (
                        row.get("request_id")
                        or row.get("RequestID")
                        or f"req-{len(traces):04d}"
                    )

                    # Parse timestamp
                    timestamp_str = (
                        row.get("arrival_timestamp_ms") or row.get("timestamp") or "0"
                    )
                    try:
                        arrival_timestamp_ms = float(timestamp_str)
                    except ValueError:
                        logger.warning(
                            f"Invalid timestamp for {request_id}: {timestamp_str}, using 0"
                        )
                        arrival_timestamp_ms = 0

                    # Parse token counts
                    try:
                        input_token_num = int(
                            row.get("input_token_num") or row.get("input_tokens") or 128
                        )
                        output_token_num = int(
                            row.get("output_token_num")
                            or row.get("output_tokens")
                            or 128
                        )
                    except ValueError:
                        logger.warning(
                            f"Invalid token numbers for {request_id}, using defaults"
                        )
                        input_token_num = 128
                        output_token_num = 128

                    # Generate the actual prompt content
                    actual_prompt = prompt_generator(input_token_num)

                    trace = RequestTrace(
                        request_id=request_id,
                        arrival_timestamp_ms=arrival_timestamp_ms,
                        input_token_num=input_token_num,
                        output_token_num=output_token_num,
                        actual_prompt=actual_prompt,
                    )
                    traces.append(trace)

            logger.info(
                f"Loaded {len(traces)} requests from CSV trace file: {trace_file}"
            )

            # Sort by timestamp
            traces.sort(key=lambda x: x.arrival_timestamp_ms)

            # Print debug information
            if traces:
                logger.info(
                    f"First request: ID={traces[0].request_id}, "
                    f"time={traces[0].arrival_timestamp_ms}ms, "
                    f"input={traces[0].input_token_num}, "
                    f"output={traces[0].output_token_num}"
                )
                if len(traces) > 1:
                    logger.info(
                        f"Last request: ID={traces[-1].request_id}, "
                        f"time={traces[-1].arrival_timestamp_ms}ms"
                    )

            return traces

        except csv.Error as e:
            logger.error(f"Failed to parse CSV file {trace_file}: {e}")
            raise ValueError(f"Invalid CSV format in {trace_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to load trace file {trace_file}: {e}")
            raise

    async def send_request(
        self, trace: RequestTrace, semaphore: asyncio.Semaphore
    ) -> RequestTrace:
        """Send a single request and record metrics"""
        # Apply arrival-time delay before execution
        if trace.arrival_timestamp_ms > 0:
            await asyncio.sleep(trace.arrival_timestamp_ms / 1000)

        # Acquire semaphore to execute the actual request
        async with semaphore:
            self.total_requests += 1

            try:
                # Record actual start time
                trace.start_time = time.perf_counter()

                # Build request payload
                payload = {
                    "model": self.config.model_name,
                    "messages": [{"role": "user", "content": trace.actual_prompt}],
                    "max_tokens": trace.output_token_num,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "stream": True,  # Use streaming response to measure TTFT
                }

                logger.debug(
                    f"Sending request {trace.request_id}: "
                    f"input={trace.input_token_num}, "
                    f"output={trace.output_token_num}"
                )

                # Send request
                async with self.session.post(
                    f"{self.config.api_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        self._handle_request_error(trace, "error", error_msg)
                        return trace

                    # Process streaming response
                    ttft, total_tokens = await self._process_stream_response(
                        response, trace
                    )

                    # Compute end-to-end latency
                    e2e_latency = (time.perf_counter() - trace.start_time) * 1000

                    # Update trace
                    trace.ttft = (
                        ttft if ttft else e2e_latency
                    )  # If no token received, use total latency as TTFT
                    trace.e2e_latency = e2e_latency
                    trace.total_tokens = total_tokens
                    trace.success = True

                    self.successful_requests += 1

                    logger.debug(
                        f"Request {trace.request_id} completed: "
                        f"TTFT={trace.ttft:.2f}ms, "
                        f"E2E={trace.e2e_latency:.2f}ms, "
                        f"tokens={total_tokens}"
                    )

            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout_ms}ms"
                self._handle_request_error(trace, "timeout", error_msg)

            except Exception as e:
                error_msg = str(e)
                self._handle_request_error(trace, "error", error_msg)

            return trace

    async def run_trace(
        self,
        traces: List[RequestTrace],
        concurrency: int = 32,
        warmup_requests: int = 10,
    ) -> Tuple[List[RequestTrace], Dict[str, Any]]:
        """Run all requests in the trace"""
        # Reset counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        logger.info(
            f"Starting trace run: {len(traces)} requests, concurrency={concurrency}"
        )

        # Warmup phase
        if warmup_requests > 0:
            logger.info(f"Warmup phase: {warmup_requests} requests")

            warmup_traces = traces[: min(warmup_requests, len(traces))]
            semaphore = asyncio.Semaphore(concurrency)
            warmup_tasks = []

            for trace in warmup_traces:
                # Warmup requests do not use arrival time intervals
                warmup_trace = RequestTrace(
                    request_id=f"warmup-{trace.request_id}",
                    arrival_timestamp_ms=0,
                    input_token_num=trace.input_token_num,
                    output_token_num=min(
                        10, trace.output_token_num
                    ),  # Generate fewer tokens during warmup
                    actual_prompt=trace.actual_prompt,
                )

                task = self.send_request(warmup_trace, semaphore)
                warmup_tasks.append(task)

            warmup_results = await asyncio.gather(*warmup_tasks, return_exceptions=True)

            # Handle exceptions
            for i, result in enumerate(warmup_results):
                if isinstance(result, Exception):
                    logger.warning(f"Warmup request {i} failed: {result}")

            logger.info("Warmup completed")

        # Measurement phase
        logger.info(f"Measurement phase: {len(traces)} requests")

        # Create semaphore to control concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

        # Record start time
        test_start_time = time.perf_counter()

        # Send all requests
        tasks = []
        for trace in traces:
            task = self.send_request(trace, self.semaphore)
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Record end time
        test_end_time = time.perf_counter()
        total_test_duration = test_end_time - test_start_time

        # Process results
        processed_traces = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Request {i} raised exception: {result}")
                # Create a failed trace record
                if i < len(traces):
                    failed_trace = traces[i]
                    failed_trace.error = str(result)
                    failed_trace.success = False
                    processed_traces.append(failed_trace)
            else:
                processed_traces.append(result)

        # Compute statistics
        stats = self._calculate_statistics(processed_traces, total_test_duration)

        stats.setdefault("total_requests", len(processed_traces))
        stats.setdefault("successful_requests", 0)
        stats.setdefault("failed_requests", len(processed_traces))
        stats.setdefault("success_rate", 0.0)
        stats.setdefault("avg_ttft", 0.0)
        stats.setdefault("avg_e2e_latency", 0.0)

        # Log summary
        logger.info(
            f"Trace run completed: "
            f"{stats['success_rate']:.2%} success rate, "
            f"{stats['avg_ttft']:.2f}ms avg TTFT, "
            f"{stats['avg_e2e_latency']:.2f}ms avg E2E latency"
        )

        return processed_traces, stats

    def _calculate_percentile_stats(
        self, values: List[float], prefix: str
    ) -> Dict[str, Any]:
        """Calculate percentile-based statistics"""
        if not values:
            return {}

        stats = {}
        stats[f"avg_{prefix}"] = np.mean(values)
        stats[f"p50_{prefix}"] = np.percentile(values, 50)
        stats[f"p95_{prefix}"] = np.percentile(values, 95)
        stats[f"p99_{prefix}"] = np.percentile(values, 99)
        stats[f"min_{prefix}"] = np.min(values)
        stats[f"max_{prefix}"] = np.max(values)
        stats[f"std_{prefix}"] = np.std(values)

        return stats

    def _calculate_statistics(
        self, traces: List[RequestTrace], total_duration: float
    ) -> Dict[str, Any]:
        """Calculate overall statistics"""
        stats = {}

        # Filter successful requests
        successful_traces = [t for t in traces if t.success]

        if not successful_traces:
            logger.warning("No successful requests in trace run")
            return stats

        # Basic statistics
        stats["total_requests"] = len(traces)
        stats["successful_requests"] = len(successful_traces)
        stats["failed_requests"] = len(traces) - len(successful_traces)
        stats["success_rate"] = len(successful_traces) / len(traces) if traces else 0

        # Time statistics
        stats["total_duration"] = total_duration
        stats["requests_per_second"] = (
            len(traces) / total_duration if total_duration > 0 else 0
        )

        # TTFT statistics
        ttfts = [t.ttft for t in successful_traces if t.ttft is not None]
        if ttfts:
            stats.update(self._calculate_percentile_stats(ttfts, "ttft"))

        # E2E latency statistics
        e2e_latencies = [
            t.e2e_latency for t in successful_traces if t.e2e_latency is not None
        ]
        if e2e_latencies:
            stats.update(self._calculate_percentile_stats(e2e_latencies, "e2e_latency"))

        # Token statistics
        total_tokens = sum(t.total_tokens for t in successful_traces)
        stats["total_tokens"] = total_tokens
        stats["avg_tokens_per_request"] = (
            total_tokens / len(successful_traces) if successful_traces else 0
        )

        # Throughput
        if total_duration > 0:
            stats["throughput_tps"] = total_tokens / total_duration  # tokens per second

        # Input/output token statistics
        input_tokens = sum(t.input_token_num for t in successful_traces)
        output_tokens = sum(t.output_token_num for t in successful_traces)
        stats["total_input_tokens"] = input_tokens
        stats["total_output_tokens"] = output_tokens

        return stats

    def save_results_to_csv(
        self, traces: List[RequestTrace], output_dir: Path, run_id: str
    ):
        """Save trace results to CSV files"""
        detailed_file = output_dir / f"{run_id}_trace_detailed.csv"
        with open(detailed_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "request_id",
                    "arrival_timestamp_ms",
                    "input_token_num",
                    "output_token_num",
                    "ttft_ms",
                    "e2e_latency_ms",
                    "total_tokens",
                    "success",
                    "error",
                ]
            )

            for trace in traces:
                writer.writerow(
                    [
                        trace.request_id,
                        trace.arrival_timestamp_ms,
                        trace.input_token_num,
                        trace.output_token_num,
                        trace.ttft if trace.ttft is not None else "",
                        trace.e2e_latency if trace.e2e_latency is not None else "",
                        trace.total_tokens,
                        trace.success,
                        trace.error or "",
                    ]
                )

        # Save time-series data
        if traces:
            # TTFT time series
            ttft_file = output_dir / f"{run_id}_trace_ttft.csv"
            with open(ttft_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["request_index", "ttft_ms"])
                for i, trace in enumerate(traces):
                    if trace.ttft is not None:
                        writer.writerow([i, trace.ttft])

            # E2E latency time series
            latency_file = output_dir / f"{run_id}_trace_latency.csv"
            with open(latency_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["request_index", "e2e_latency_ms"])
                for i, trace in enumerate(traces):
                    if trace.e2e_latency is not None:
                        writer.writerow([i, trace.e2e_latency])

        logger.info(f"Trace results saved to {output_dir}")
        return detailed_file

# utils/trace_client.py
#!/usr/bin/env python3
"""
Trace client implementation
Reads requests from a trace file and sends them to the inference service according to time intervals.
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

logger = logging.getLogger(__name__)

@dataclass
class RequestTrace:
    """Single request entry in the trace file"""
    request_id: str
    arrival_timestamp_ms: float  # timestamp in milliseconds (relative or absolute)
    input_token_num: int
    output_token_num: int
    
    # runtime computed fields
    actual_prompt: Optional[str] = None
    start_time: Optional[float] = None
    ttft: Optional[float] = None  # Time To First Token (ms)
    e2e_latency: Optional[float] = None  # End-to-end latency (ms)
    total_tokens: int = 0
    success: bool = False
    error: Optional[str] = None

@dataclass
class TraceClientConfig:
    """Trace client configuration"""
    api_url: str
    model_name: str
    timeout_ms: int = 30000
    max_retries: int = 3
    warmup_requests: int = 10

class TraceClient:
    """Trace client"""
    
    def __init__(self, config: TraceClientConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_counter = 0
        self.semaphore: Optional[asyncio.Semaphore] = None
        
        # performance statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_ms/1000 + 10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    @staticmethod
    def load_trace_file(trace_file: str, prompt_generator) -> List[RequestTrace]:
       """
        Load trace file
        
        Args:
            trace_file: Path to the trace file
            prompt_generator: Function to generate actual prompts
            
        Returns:
            List of request traces
        """
        traces = []
        
        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                # Auto detect file format
                first_line = f.readline().strip()
                f.seek(0)
                
                if trace_file.endswith('.csv') or ',' in first_line:
                    # CSV format
                    reader = csv.DictReader(f)
                    for row in reader:
                        # support variations of column names
                        request_id = row.get('request_id') or row.get('RequestID') or f"req-{len(traces):04d}"
                        
                        # parse timestamp
                        timestamp_str = row.get('arrival_timestamp_ms') or row.get('timestamp') or '0'
                        try:
                            arrival_timestamp_ms = float(timestamp_str)
                        except ValueError:
                            logger.warning(f"Invalid timestamp for {request_id}: {timestamp_str}, using 0")
                            arrival_timestamp_ms = 0
                        
                        # parse token counts
                        try:
                            input_token_num = int(row.get('input_token_num') or row.get('input_tokens') or 128)
                            output_token_num = int(row.get('output_token_num') or row.get('output_tokens') or 128)
                        except ValueError:
                            logger.warning(f"Invalid token numbers for {request_id}, using defaults")
                            input_token_num = 128
                            output_token_num = 128
                        
                        # generate prompt content
                        actual_prompt = prompt_generator(input_token_num)
                        
                        trace = RequestTrace(
                            request_id=request_id,
                            arrival_timestamp_ms=arrival_timestamp_ms,
                            input_token_num=input_token_num,
                            output_token_num=output_token_num,
                            actual_prompt=actual_prompt
                        )
                        traces.append(trace)
                
                elif trace_file.endswith('.json') or first_line.startswith('[') or first_line.startswith('{'):
                    # JSON format
                    data = json.load(f)
                    
                    if isinstance(data, dict):
                        data = [data]
                    
                    for i, item in enumerate(data):
                        request_id = item.get('request_id') or f"req-{i:04d}"
                        
                        arrival_timestamp_ms = float(item.get('arrival_timestamp_ms', 0))
                        input_token_num = int(item.get('input_token_num', 128))
                        output_token_num = int(item.get('output_token_num', 128))
                        
                        actual_prompt = prompt_generator(input_token_num)
                        
                        trace = RequestTrace(
                            request_id=request_id,
                            arrival_timestamp_ms=arrival_timestamp_ms,
                            input_token_num=input_token_num,
                            output_token_num=output_token_num,
                            actual_prompt=actual_prompt
                        )
                        traces.append(trace)
                
                else:
                    raise ValueError(f"Unsupported trace file format: {trace_file}")
            
            logger.info(f"Loaded {len(traces)} requests from trace file: {trace_file}")
            
            # sort by timestamp
            traces.sort(key=lambda x: x.arrival_timestamp_ms)
            
            # print debug info for first few entries
            if traces:
                logger.info(f"First request: ID={traces[0].request_id}, "
                          f"time={traces[0].arrival_timestamp_ms}ms, "
                          f"input={traces[0].input_token_num}, "
                          f"output={traces[0].output_token_num}")
                if len(traces) > 1:
                    logger.info(f"Last request: ID={traces[-1].request_id}, "
                              f"time={traces[-1].arrival_timestamp_ms}ms")
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to load trace file {trace_file}: {e}")
            raise
    
    async def send_request(
        self, 
        trace: RequestTrace, 
        semaphore: asyncio.Semaphore
    ) -> RequestTrace:
        """
        Send a single request and record metrics
        
        Args:
            trace: Request trace
            semaphore: Concurrency control semaphore
            
        Returns:
            Updated request trace with performance metrics
        """
        async with semaphore:
            self.total_requests += 1
            request_start = time.perf_counter()
            ttft = None
            total_tokens = 0
            
            try:
                 # Simulate request arrival timing (assuming timestamps are relative)
                if trace.arrival_timestamp_ms > 0:
                    await asyncio.sleep(trace.arrival_timestamp_ms / 1000)
                
                # record start time
                trace.start_time = time.perf_counter()
                
                # construct request payload
                payload = {
                    "model": self.config.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": trace.actual_prompt
                        }
                    ],
                    "max_tokens": trace.output_token_num,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": True   # use streaming to measure TTFT
                }
                
                logger.debug(f"Sending request {trace.request_id}: "
                           f"input={trace.input_token_num}, "
                           f"output={trace.output_token_num}")
                
                # send request
                async with self.session.post(
                    f"{self.config.api_url}/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {await response.text()}"
                        trace.error = error_msg
                        trace.success = False
                        self.failed_requests += 1
                        
                        logger.error(f"Request {trace.request_id} failed: {error_msg}")
                        return trace
                    
                    # process streaming response
                    first_token_received = False
                    first_token_time = None
                    
                    async for line in response.content:
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data_str = line[6:]
                                
                                if data_str == '[DONE]':
                                    break
                                
                                try:
                                    data = json.loads(data_str)
                                    if 'choices' in data and len(data['choices']) > 0:
                                        choice = data['choices'][0]
                                        
                                        # check for content
                                        if 'delta' in choice and 'content' in choice['delta']:
                                            content = choice['delta']['content']
                                            
                                            if not first_token_received:
                                                first_token_time = time.perf_counter()
                                                ttft = (first_token_time - trace.start_time) * 1000
                                                first_token_received = True
                                                logger.debug(f"First token received for {trace.request_id}: "
                                                           f"TTFT={ttft:.2f}ms")
                                            
                                            # count tokens (rough estimate)
                                            if content.strip():
                                                total_tokens += 1
                                        
                                        # check finish condition
                                        if choice.get('finish_reason'):
                                            break
                                            
                                except json.JSONDecodeError:
                                    logger.warning(f"Invalid JSON in stream: {line}")
                                    continue
                    
                    # compute e2e latency
                    e2e_latency = (time.perf_counter() - trace.start_time) * 1000
                    
                    # update trace
                    trace.ttft = ttft if ttft else e2e_latency  
                    trace.e2e_latency = e2e_latency
                    trace.total_tokens = total_tokens
                    trace.success = True
                    
                    self.successful_requests += 1
                    
                    logger.debug(f"Request {trace.request_id} completed: "
                               f"TTFT={trace.ttft:.2f}ms, "
                               f"E2E={trace.e2e_latency:.2f}ms, "
                               f"tokens={total_tokens}")
                
            except asyncio.TimeoutError:
                error_msg = f"Timeout after {self.config.timeout_ms}ms"
                trace.error = error_msg
                trace.success = False
                self.failed_requests += 1
                logger.error(f"Request {trace.request_id} timeout: {error_msg}")
                
            except Exception as e:
                error_msg = str(e)
                trace.error = error_msg
                trace.success = False
                self.failed_requests += 1
                logger.error(f"Request {trace.request_id} failed: {error_msg}")
            
            return trace
    
    async def run_trace(
        self, 
        traces: List[RequestTrace], 
        concurrency: int = 32,
        warmup_requests: int = 10
    ) -> Tuple[List[RequestTrace], Dict[str, Any]]:
        """
        Run all requests in the trace
        
        Args:
            traces: List of request traces
            concurrency: Max concurrency
            warmup_requests: Number of warmup requests
            
        Returns:
            (updated traces, statistics dictionary)
        """
        # reset counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info(f"Starting trace run: {len(traces)} requests, concurrency={concurrency}")
        
        # warmup phase
        if warmup_requests > 0:
            logger.info(f"Warmup phase: {warmup_requests} requests")
            
            warmup_traces = traces[:min(warmup_requests, len(traces))]
            semaphore = asyncio.Semaphore(concurrency)
            warmup_tasks = []
            
            for trace in warmup_traces:
                # Warm-up requests do not calculate arrival interval
                warmup_trace = RequestTrace(
                    request_id=f"warmup-{trace.request_id}",
                    arrival_timestamp_ms=0,
                    input_token_num=trace.input_token_num,
                    output_token_num=min(10, trace.output_token_num), # Warm-up generates fewer tokens
                    actual_prompt=trace.actual_prompt
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
                if i < len(traces):
                    failed_trace = traces[i]
                    failed_trace.error = str(result)
                    failed_trace.success = False
                    processed_traces.append(failed_trace)
            else:
                processed_traces.append(result)
        
        # Calculate statistics
        stats = self._calculate_statistics(processed_traces, total_test_duration)
        
        # Log summary
        logger.info(f"Trace run completed: "
                   f"{stats['success_rate']:.2%} success rate, "
                   f"{stats['avg_ttft']:.2f}ms avg TTFT, "
                   f"{stats['avg_e2e_latency']:.2f}ms avg E2E latency")
        
        return processed_traces, stats
    
    def _calculate_statistics(
        self, 
        traces: List[RequestTrace], 
        total_duration: float
    ) -> Dict[str, Any]:
        stats = {}
        
        # Filter successful requests
        successful_traces = [t for t in traces if t.success]
        
        if not successful_traces:
            logger.warning("No successful requests in trace run")
            return stats
        
        # Basic statistics
        stats['total_requests'] = len(traces)
        stats['successful_requests'] = len(successful_traces)
        stats['failed_requests'] = len(traces) - len(successful_traces)
        stats['success_rate'] = len(successful_traces) / len(traces) if traces else 0
        
        #  Time statistics
        stats['total_duration'] = total_duration
        stats['requests_per_second'] = len(traces) / total_duration if total_duration > 0 else 0
        
        # TTFT statistics
        ttfts = [t.ttft for t in successful_traces if t.ttft is not None]
        if ttfts:
            stats['avg_ttft'] = np.mean(ttfts)
            stats['p50_ttft'] = np.percentile(ttfts, 50)
            stats['p95_ttft'] = np.percentile(ttfts, 95)
            stats['p99_ttft'] = np.percentile(ttfts, 99)
            stats['min_ttft'] = np.min(ttfts)
            stats['max_ttft'] = np.max(ttfts)
            stats['std_ttft'] = np.std(ttfts)
        
        # E2E latency statistics
        e2e_latencies = [t.e2e_latency for t in successful_traces if t.e2e_latency is not None]
        if e2e_latencies:
            stats['avg_e2e_latency'] = np.mean(e2e_latencies)
            stats['p50_e2e_latency'] = np.percentile(e2e_latencies, 50)
            stats['p95_e2e_latency'] = np.percentile(e2e_latencies, 95)
            stats['p99_e2e_latency'] = np.percentile(e2e_latencies, 99)
            stats['min_e2e_latency'] = np.min(e2e_latencies)
            stats['max_e2e_latency'] = np.max(e2e_latencies)
            stats['std_e2e_latency'] = np.std(e2e_latencies)
        
        # Token statistics
        total_tokens = sum(t.total_tokens for t in successful_traces)
        stats['total_tokens'] = total_tokens
        stats['avg_tokens_per_request'] = total_tokens / len(successful_traces) if successful_traces else 0
        
        # Throughput
        if total_duration > 0:
            stats['throughput_tps'] = total_tokens / total_duration  # tokens per second
        
        # Input/output token statistics
        input_tokens = sum(t.input_token_num for t in successful_traces)
        output_tokens = sum(t.output_token_num for t in successful_traces)
        stats['total_input_tokens'] = input_tokens
        stats['total_output_tokens'] = output_tokens
        
        return stats
    
    def save_results_to_csv(
        self, 
        traces: List[RequestTrace], 
        output_dir: Path, 
        run_id: str
    ):
        # Save detailed results
        detailed_file = output_dir / f"{run_id}_trace_detailed.csv"
        with open(detailed_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'request_id', 'arrival_timestamp_ms', 'input_token_num',
                'output_token_num', 'ttft_ms', 'e2e_latency_ms',
                'total_tokens', 'success', 'error'
            ])
            
            for trace in traces:
                writer.writerow([
                    trace.request_id,
                    trace.arrival_timestamp_ms,
                    trace.input_token_num,
                    trace.output_token_num,
                    trace.ttft if trace.ttft is not None else '',
                    trace.e2e_latency if trace.e2e_latency is not None else '',
                    trace.total_tokens,
                    trace.success,
                    trace.error or ''
                ])
        
        # Save time-series files (for plotting)
        if traces:
            # TTFT time series
            ttft_file = output_dir / f"{run_id}_trace_ttft.csv"
            with open(ttft_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['request_index', 'ttft_ms'])
                for i, trace in enumerate(traces):
                    if trace.ttft is not None:
                        writer.writerow([i, trace.ttft])
            
            # E2E latency time series
            latency_file = output_dir / f"{run_id}_trace_latency.csv"
            with open(latency_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['request_index', 'e2e_latency_ms'])
                for i, trace in enumerate(traces):
                    if trace.e2e_latency is not None:
                        writer.writerow([i, trace.e2e_latency])
        
        logger.info(f"Trace results saved to {output_dir}")
        return detailed_file

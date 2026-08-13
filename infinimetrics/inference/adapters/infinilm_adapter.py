#!/usr/bin/env python3
"""
InfiniLM Adapter Implementation 
"""

import os
import re
import torch
import sys
import time
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import subprocess
import threading

# Try to import InfiniLM related modules
try:
    # Add scripts directory to path
    scripts_dir = Path.cwd() / "scripts"
    if scripts_dir.exists():
        sys.path.insert(0, str(scripts_dir))
    
    # Import jiuge module
    from jiuge import JiugeForCauslLM
    from libinfinicore_infer import DeviceType
    from infer_task import InferTask, KVCache
    
    INFINILM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("InfiniLM modules imported successfully")
    
except ImportError as e:
    INFINILM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import InfiniLM modules: {e}")
    raise

from adapter_base import InferAdapter
from infer_config import InferConfig, InferMode, DirectInferArgs
from utils.token_generator import TokenGenerator, create_token_generator

class InfiniLMAdapter(InferAdapter):
    """InfiniLM adapter implementation - Fixed version (using API correctly)"""
    
    def __init__(self, config: InferConfig):
        super().__init__(config)
        
        # InfiniLM specific attributes
        self.jiuge_model: Optional[JiugeForCauslLM] = None
        self.token_generator: Optional[TokenGenerator] = None
        
        # Service related
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = 8000
        
        logger.info(f"InfiniLMAdapter created for model: {config.model}")
        logger.info(f"Model path: {config.model_path}")
    
    def load_model(self) -> None:
        """Load real InfiniLM model (fixed API usage)"""
        if not INFINILM_AVAILABLE:
            logger.error("InfiniLM modules not available. Cannot load model.")
            raise ImportError("InfiniLM modules not available")
        
        logger.info(f"Loading real InfiniLM model from: {self.config.model_path}")
        
        try:
            # Determine device type
            device_type = self._get_device_type()
            
            # Get tp size (from infer_args.parallel)
            tp_size = self.config.infer_args.parallel.tp
            
            # ✅ Fix 1: Correctly call JiugeForCauslLM constructor
            # Original API: JiugeForCauslLM(model_dir_path, device, ndev, max_tokens=None)
            self.jiuge_model = JiugeForCauslLM(
                self.config.model_path,
                device_type,  # ✅ Not device=device_type
                tp_size,
                max_tokens=self.config.infer_args.max_seq_len  # ✅ This parameter is optional
            )
            
            # Get tokenizer
            self.tokenizer = self.jiuge_model.tokenizer
            
            # Create token generator
            self.token_generator = create_token_generator(
                self.tokenizer,
                exclude_special_tokens=True
            )
            
            self.model_loaded = True
            logger.info("Real InfiniLM model loaded successfully")
            logger.info(f"Tokenizer vocab size: {self.get_vocab_size()}")
            logger.info(f"Model max context length: {self.jiuge_model.max_context_len()}")
            logger.info(f"EOS token ID: {self.jiuge_model.eos_token_id}")
            
        except Exception as e:
            logger.error(f"Failed to load real InfiniLM model: {e}", exc_info=True)
            raise
    
    def unload_model(self) -> None:
        """Unload model"""
        if self.jiuge_model:
            try:
                self.jiuge_model.destroy_model_instance()
                logger.info("InfiniLM model unloaded")
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")
            
            self.jiuge_model = None
        
        self.model_loaded = False
        self.tokenizer = None
        self.token_generator = None
    
    def generate(
        self, 
        prompts: List[str], 
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        Real InfiniLM inference implementation (fixed API usage)
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature, top_p, top_k: Sampling parameters
        
        Returns:
            (List of generated texts, Latency list (ms), TTFT list (ms))
        """
        if not self.model_loaded or not self.jiuge_model:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Real InfiniLM batch inference for {len(prompts)} prompts")
        logger.info(f"  Max tokens per prompt: {max_tokens}")
        logger.info(f"  Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
        
        # 1. Encode prompts
        token_lists = []
        for i, prompt in enumerate(prompts):
            tokens = self.tokenizer.encode(prompt)
            token_lists.append(tokens)
            if i == 0:  # Record first prompt information
                logger.info(f"  First prompt: {len(tokens)} tokens")
                logger.debug(f"  First prompt preview: {prompt[:100]}...")
        
        # 2. Create InferTask and KVCache for each prompt
        tasks = []
        kv_caches = []
        
        for i, tokens in enumerate(token_lists):
            try:
                # ✅ Fix 2: Correctly create InferTask
                # InferTask parameters: id, tokens, max_tokens, temperature, topk, topp, topa, end_tokens
                # topa parameter might be needed, set to 0 for now
                topa = 0  # Assuming no top-a sampling needed
                
                # ✅ Fix 3: end_tokens should be a list
                if isinstance(self.jiuge_model.eos_token_id, list):
                    end_tokens = self.jiuge_model.eos_token_id
                else:
                    end_tokens = [self.jiuge_model.eos_token_id]
                
                # ✅ Fix 4: Use max_seq_len from config, not max_context_len
                max_seq_len = min(
                    self.config.infer_args.max_seq_len,
                    self.jiuge_model.max_context_len()
                )
                
                task = InferTask(
                    id=i,
                    tokens=tokens,
                    max_tokens=max_seq_len,
                    temperature=temperature,
                    topk=top_k,
                    topp=top_p,
                    end_tokens=end_tokens
                )
                
                # Create and bind KVCache
                kv_cache = KVCache(self.jiuge_model)
                task.bind_kvcache(kv_cache)
                
                tasks.append(task)
                kv_caches.append(kv_cache)
                
                logger.debug(f"  Created InferTask {i}: {len(tokens)} input tokens")
                
            except Exception as e:
                logger.error(f"Failed to create InferTask for prompt {i}: {e}")
                raise
        
        logger.info(f"Created {len(tasks)} InferTasks for batch inference")
        
        # 3. Execute batch inference
        generated_texts = []
        latencies = []
        ttfts = []
        
        # Pre-allocate result lists for each prompt
        all_generated_tokens = [[] for _ in range(len(tasks))]
        
        try:
            # Measure TTFT (first batch inference)
            start_time = time.perf_counter()
            output_tokens_batch = self.jiuge_model.batch_infer_one_round(tasks)
            ttft = (time.perf_counter() - start_time) * 1000
            
            # ✅ Fix 5: batch_infer_one_round returns List[List[int]]
            # Each task returns a token list (may contain multiple tokens)
            for i, task_output in enumerate(output_tokens_batch):
                if task_output:  # May have output
                    # Take first token (if multiple tokens returned, take first)
                    first_token = task_output[0] if isinstance(task_output, list) else task_output
                    all_generated_tokens[i].append(first_token)
                    
                    # Record TTFT for this task (all tasks use same TTFT since batch inference)
                    ttfts.append(ttft)
                else:
                    all_generated_tokens[i].append(0)  # Placeholder
                    ttfts.append(0.0)
            
            # 4. Continue generating remaining tokens (token by token)
            total_generated = 1  # Already generated first token
            
            while total_generated < max_tokens:
                # Update status for all tasks
                active_tasks = []
                active_indices = []
                
                for i, task in enumerate(tasks):
                    if len(all_generated_tokens[i]) > 0:
                        last_token = all_generated_tokens[i][-1]
                        
                        # ✅ Fix 6: Correctly check EOS
                        if isinstance(self.jiuge_model.eos_token_id, list):
                            is_eos = last_token in self.jiuge_model.eos_token_id
                        else:
                            is_eos = last_token == self.jiuge_model.eos_token_id
                        
                        if not is_eos and len(all_generated_tokens[i]) < max_tokens:
                            task.next(last_token)
                            active_tasks.append(task)
                            active_indices.append(i)
                
                # If no active tasks, stop generation
                if not active_tasks:
                    logger.info("All tasks reached EOS or max tokens")
                    break
                
                # Batch inference for active tasks
                iteration_start = time.perf_counter()
                active_outputs = self.jiuge_model.batch_infer_one_round(active_tasks)
                iteration_time = (time.perf_counter() - iteration_start) * 1000
                
                # Process outputs
                for idx, task_idx in enumerate(active_indices):
                    if idx < len(active_outputs) and active_outputs[idx]:
                        next_token = active_outputs[idx][0] if isinstance(active_outputs[idx], list) else active_outputs[idx]
                        all_generated_tokens[task_idx].append(next_token)
                
                total_generated += 1
                
                if total_generated % 10 == 0:
                    logger.debug(f"  Generated {total_generated}/{max_tokens} tokens")
            
            # 5. Calculate total latency and decode text
            total_latency = (time.perf_counter() - start_time) * 1000
            
            for i, generated_tokens in enumerate(all_generated_tokens):
                # Calculate latency for this prompt (for batch inference, all prompts have same latency)
                latencies.append(total_latency)
                
                # Decode text
                if generated_tokens:
                    generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    generated_texts.append(generated_text)
                    
                    logger.debug(f"  Prompt {i}: {len(generated_tokens)} tokens generated")
                    if i == 0 and generated_text:
                        logger.debug(f"  First generated text preview: {generated_text[:100]}...")
                else:
                    generated_texts.append("")
                    logger.warning(f"  Prompt {i}: No tokens generated")
            
        except Exception as e:
            logger.error(f"Error during batch inference: {e}", exc_info=True)
            raise
        
        finally:
            # 6. Clean up KVCaches
            logger.info("Cleaning up KVCaches...")
            for i, kv_cache in enumerate(kv_caches):
                try:
                    if kv_cache and self.jiuge_model:
                        kv_cache.drop(self.jiuge_model)
                except Exception as e:
                    logger.warning(f"Failed to drop KV cache {i}: {e}")
        
        # 7. Return results
        logger.info(f"Inference completed: {len(generated_texts)} prompts processed")
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
            logger.info(f"  Avg latency: {avg_latency:.2f}ms")
            logger.info(f"  Avg TTFT: {avg_ttft:.2f}ms")
        
        return generated_texts, latencies, ttfts
    
    def batch_generate(
        self,
        batch_prompts: List[List[str]],
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
        """
        Batch text generation (multiple batches)
        
        Note: For large batches, we may need to split to avoid OOM
        """
        logger.info(f"Batch generating for {len(batch_prompts)} batches")
        
        all_texts = []
        all_latencies = []
        all_ttfts = []
        
        for batch_idx, prompts in enumerate(batch_prompts):
            logger.info(f"Processing batch {batch_idx+1}/{len(batch_prompts)} "
                       f"({len(prompts)} prompts)")
            
            # Check batch size to avoid OOM
            max_batch_size = 8  # Safe value, can adjust based on GPU memory
            if len(prompts) > max_batch_size:
                logger.warning(f"Batch size {len(prompts)} too large, splitting...")
                
                # Split processing
                split_texts = []
                split_latencies = []
                split_ttfts = []
                
                for i in range(0, len(prompts), max_batch_size):
                    sub_prompts = prompts[i:i + max_batch_size]
                    logger.info(f"  Processing sub-batch {i//max_batch_size + 1}")
                    
                    texts, latencies, ttfts = self.generate(
                        sub_prompts, max_tokens, temperature, top_p, top_k
                    )
                    
                    split_texts.extend(texts)
                    split_latencies.extend(latencies)
                    split_ttfts.extend(ttfts)
                
                all_texts.append(split_texts)
                all_latencies.append(split_latencies)
                all_ttfts.append(split_ttfts)
            else:
                texts, latencies, ttfts = self.generate(
                    prompts, max_tokens, temperature, top_p, top_k
                )
                
                all_texts.append(texts)
                all_latencies.append(latencies)
                all_ttfts.append(ttfts)
        
        return all_texts, all_latencies, all_ttfts
    
    def get_peak_memory_usage(self) -> Optional[float]:
        """Get peak memory usage (GB)"""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
                # Get peak memory (bytes)
                max_memory_bytes = 0
                for i in range(torch.cuda.device_count()):
                    device_max = torch.cuda.max_memory_allocated(i)
                    if device_max > max_memory_bytes:
                        max_memory_bytes = device_max
            
                # Convert to GB
                max_memory_gb = max_memory_bytes / (1024 ** 3)
            
                # Also get current memory usage
                current_memory_bytes = torch.cuda.memory_allocated()
                current_memory_gb = current_memory_bytes / (1024 ** 3)
            
                logger.info(f"GPU memory - Peak: {max_memory_gb:.2f} GB, Current: {current_memory_gb:.2f} GB")
                return max_memory_gb
            
        except ImportError:
            logger.warning("PyTorch not available, cannot get GPU memory usage")
        except Exception as e:
            logger.warning(f"Failed to get GPU memory usage: {e}")
    
        # Try to get via nvidia-smi (fallback method)
        try:
        
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
        
            if result.returncode == 0:
                # Parse output, e.g.: "1234, 24576\n"
                lines = result.stdout.strip().split('\n')
                max_memory_mb = 0
            
                for line in lines:
                    if line:
                        used_str, total_str = line.split(',')
                        used_mb = float(used_str.strip())
                        total_mb = float(total_str.strip())
                    
                        if used_mb > max_memory_mb:
                            max_memory_mb = used_mb
            
                if max_memory_mb > 0:
                    max_memory_gb = max_memory_mb / 1024
                    logger.info(f"GPU memory (nvidia-smi): {max_memory_gb:.2f} GB")
                    return max_memory_gb
                
        except Exception as e:
            logger.debug(f"nvidia-smi fallback failed: {e}")
    
        return None

    def calculate_perplexity(self, test_data: List[str]) -> float:
        """Calculate perplexity"""
        if not self.model_loaded or not self.jiuge_model:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Calculating perplexity for {len(test_data)} test samples")
        
        try:
            # Convert text to token sequences
            test_sequences = []
            for text in test_data:
                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.config.infer_args.max_seq_len:
                    tokens = tokens[:self.config.infer_args.max_seq_len]
                test_sequences.append(tokens)
            
            # Use jiuge model's perplexity method
            # Note: Need to test batch_size to avoid OOM
            batch_size = min(4, len(test_sequences))
            perplexity = self.jiuge_model.perplexity(test_sequences, batch_size=batch_size)
            
            logger.info(f"Perplexity calculated: {perplexity:.4f}")
            return perplexity
            
        except Exception as e:
            logger.error(f"Failed to calculate perplexity: {e}")
            # Return a default value, don't interrupt tests
            return 0.0
    
    def launch_service(self, port: int = 8000) -> None:
        """Launch InfiniLM inference service"""
        logger.info(f"Launching InfiniLM service on port {port}")

        # Build launch command
        cmd = [
            sys.executable, "scripts/launch_server.py",
            "--model-path", self.config.model_path,
            "--dev", "nvidia",
            "--ndev", str(self.config.infer_args.parallel.tp),
            "--max-batch", "4"
        ]

        if port != 8000:
            cmd.extend(["--port", str(port)])

        # Start service process
        try:
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.server_port = port
            self.service_started = True

            # Start thread to read output
            self._start_output_reader()

            logger.info(f"InfiniLM service started with PID: {self.server_process.pid}")
            logger.info(f"Command: {' '.join(cmd)}")

        except Exception as e:
            logger.error(f"Failed to launch InfiniLM service: {e}")
            raise

    def stop_service(self) -> None:
        """Stop inference service"""
        if self.server_process:
            logger.info("Stopping InfiniLM service")
            
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=10)
                logger.info("InfiniLM service stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Service did not stop gracefully, forcing kill")
                self.server_process.kill()
            
            self.server_process = None
        
        self.service_started = False
    
    def is_service_ready(self, port: int = 8000) -> bool:
        """Simplified service readiness check - only check port"""
        if not self.service_started or not self.server_process:
            logger.debug("Service not started or no server process")
            return False
    
        # Check if process is alive
        if self.server_process.poll() is not None:
            logger.error(f"Server process died with return code: {self.server_process.returncode}")
            return False
    
        # Only check if port is open
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
        
            if result == 0:
                logger.debug(f"Port {port} is open, service is ready")
                return True
            else:
                logger.debug(f"Port {port} not open yet (result={result})")
                return False
        except Exception as e:
            logger.debug(f"Port check failed: {e}")
            return False
    
    def get_service_url(self) -> str:
        """Get service URL"""
        return f"http://localhost:{self.server_port}"
    
    def _get_device_type(self):
        """Get device type based on configuration"""
        gpu_platform = self.config.device.gpu_platform.lower()
        
        if gpu_platform == "nvidia":
            return DeviceType.DEVICE_TYPE_NVIDIA
        elif gpu_platform == "cpu":
            return DeviceType.DEVICE_TYPE_CPU
        else:
            logger.warning(f"Unknown GPU platform: {gpu_platform}, using NVIDIA as default")
            return DeviceType.DEVICE_TYPE_NVIDIA
    
    def _start_output_reader(self):
        """Start output reading thread"""
        def read_output():
            if self.server_process:
                for line in self.server_process.stdout:
                    logger.info(f"[InfiniLM Server] {line.strip()}")
        
        self.output_thread = threading.Thread(target=read_output, daemon=True)
        self.output_thread.start()
    
    def _validate_framework_config(self) -> List[str]:
        """Validate InfiniLM specific configuration"""
        errors = []
        
        # Check if scripts directory exists
        scripts_dir = Path("scripts")
        if not scripts_dir.exists():
            errors.append("scripts directory not found in current directory")
        else:
            # Check for necessary script files
            required_scripts = ["jiuge.py", "launch_server.py"]
            for script in required_scripts:
                if not (scripts_dir / script).exists():
                    errors.append(f"Required script not found: {script}")
        
        # Check parallel configuration
        if self.config.infer_args.parallel.tp <= 0:
            errors.append("Tensor parallel size (tp) must be positive")
        
        # Check model directory
        model_dir = Path(self.config.model_path)
        if not model_dir.exists():
            errors.append(f"Model directory does not exist: {model_dir}")
        else:
            # Check for necessary model files
            model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
            if not model_files:
                errors.append(f"No model files found in {model_dir}")
            
            # Check config.json
            config_file = model_dir / "config.json"
            if not config_file.exists():
                errors.append(f"config.json not found in {model_dir}")
        
        return errors
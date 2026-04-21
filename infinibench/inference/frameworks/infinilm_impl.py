#!/usr/bin/env python3
"""
InfiniLM Implementation
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
from infinibench.inference.infer_config import InferConfig
import subprocess
import threading

from infinibench.common.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)


def _locate_infinilm_scripts(user_path: Optional[str] = None) -> Optional[Path]:
    """
    Inline function: locate InfiniLM scripts directory
    Args:
        user_path: user-specified path
    Returns:
        Path object or None
    """
    # Search priority
    search_paths = []

    # User-specified path
    if user_path:
        search_paths.append(Path(user_path))

    # Environment variable
    env_path = os.environ.get("INFINILM_PATH")
    if env_path:
        search_paths.append(Path(env_path))

    # Current working directory
    search_paths.append(Path.cwd())

    # Common locations
    search_paths.extend(
        [
            Path.home() / "InfiniLM",
            Path("/opt/InfiniLM"),
            Path("/usr/local/InfiniLM"),
        ]
    )

    # Search
    for base_path in search_paths:
        scripts_dir = base_path / "scripts"
        if scripts_dir.exists():
            logger.info(f"Found InfiniLM scripts at: {scripts_dir}")
            return scripts_dir

    logger.error("Cannot locate InfiniLM scripts directory")
    return None


# Attempt to import InfiniLM-related modules
INFINILM_AVAILABLE = False
JiugeForCauslLM = None
DeviceType = None
InferTask = None
KVCache = None

try:
    # First try importing from standard package
    try:
        from infinilm.jiuge import JiugeForCauslLM
        from infinilm.libinfinicore_infer import DeviceType as InfiniDeviceType
        from infinilm.infer_task import InferTask, KVCache

        INFINILM_AVAILABLE = True
        InfiniLMModel = JiugeForCauslLM
        DeviceType = InfiniDeviceType
        logger.info("InfiniLM modules imported from package")

    except ImportError:
        scripts_dir = _locate_infinilm_scripts()

        if scripts_dir and (scripts_dir / "jiuge.py").exists():
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))

            from jiuge import JiugeForCauslLM
            from libinfinicore_infer import DeviceType as InfiniDeviceType
            from infer_task import InferTask, KVCache

            INFINILM_AVAILABLE = True
            DeviceType = InfiniDeviceType
            logger.info("InfiniLM modules imported from scripts directory")
        else:
            raise ImportError("Cannot locate InfiniLM scripts directory")

except ImportError as e:
    INFINILM_AVAILABLE = False
    logger.error(f"Failed to import InfiniLM modules: {e}")


class InfiniLMImpl:
    """InfiniLM implementation"""

    def __init__(self, config):
        self.config = config

        # InfiniLM specific attributes
        self.model_instance: Optional[JiugeForCauslLM] = None
        self.model_loaded = False
        self.tokenizer = None
        self.eos_token_id = None

        logger.info(f"InfiniLMImpl created for model: {config.model}")
        logger.info(f"Model path: {config.model_path}")

    def load_model(self) -> None:
        """Load real InfiniLM model"""
        if not INFINILM_AVAILABLE or JiugeForCauslLM is None:
            logger.error("InfiniLM modules not available. Cannot load model.")
            raise ImportError("InfiniLM modules not available")

        logger.info(f"Loading InfiniLM model from: {self.config.model_path}")

        try:
            # Determine device type
            device_type = self._get_device_type()

            # Get tp size
            tp_size = 1
            infer_args = self.config.infer_args

            if isinstance(infer_args, dict):
                # processing dictionary format
                parallel_config = infer_args.get("parallel", {})
                if isinstance(parallel_config, dict):
                    tp_size = parallel_config.get("tp", 1)
                elif hasattr(parallel_config, "tp"):
                    tp_size = parallel_config.tp
            else:
                # processing object format
                if hasattr(infer_args, "parallel") and hasattr(
                    infer_args.parallel, "tp"
                ):
                    tp_size = infer_args.parallel.tp

            # Get max_seq_len
            max_seq_len = DEFAULT_MAX_SEQ_LEN
            if isinstance(infer_args, dict):
                max_seq_len = infer_args.get("max_seq_len", DEFAULT_MAX_SEQ_LEN)
            elif hasattr(infer_args, "max_seq_len"):
                max_seq_len = infer_args.max_seq_len

            logger.info(f"Using tp_size={tp_size}, max_seq_len={max_seq_len}")

            # Correctly call JiugeForCauslLM constructor
            self.model_instance = JiugeForCauslLM(
                self.config.model_path,
                device_type,
                tp_size,
                max_tokens=max_seq_len,  # optional
            )

            # Get tokenizer
            self.tokenizer = self.model_instance.tokenizer

            self.model_loaded = True
            logger.info("Real InfiniLM model loaded successfully")
            logger.info(f"Tokenizer vocab size: {self.get_vocab_size()}")

        except Exception as e:
            logger.error(f"Failed to load InfiniLM model: {e}", exc_info=True)
            raise

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return len(self.tokenizer)

    def get_special_token_ids(self) -> set:
        """Get a set of special token IDs that should be excluded"""
        if self.tokenizer is None:
            return set()

        special_ids = set()
        token_attrs = ["bos_token", "eos_token", "pad_token", "unk_token"]

        for attr in token_attrs:
            token = getattr(self.tokenizer, attr, None)
            if token and isinstance(token, str):
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                    if token_id is not None:
                        special_ids.add(token_id)
                except:
                    pass

        logger.debug(f"Found {len(special_ids)} special token IDs")
        return special_ids

    def generate_random_tokens(
        self, num_tokens: int, exclude_special: bool = True
    ) -> List[int]:
        """
        Generate a sequence of random token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")

        vocab_size = self.get_vocab_size()

        if exclude_special:
            special_ids = self.get_special_token_ids()
            all_ids = set(range(vocab_size))
            valid_ids = sorted(list(all_ids - special_ids))

            if not valid_ids:
                logger.warning(
                    "No valid tokens after excluding special tokens, using all tokens"
                )
                valid_ids = list(range(vocab_size))
        else:
            valid_ids = list(range(vocab_size))

        tokens = random.choices(valid_ids, k=num_tokens)

        logger.debug(
            f"Generated {num_tokens} random tokens (vocab_size={vocab_size}, "
            f"exclude_special={exclude_special})"
        )

        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text (for debugging)"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def validate_config(self) -> List[str]:
        """Validate configuration"""
        errors = []
        if not INFINILM_AVAILABLE:
            errors.append("InfiniLM modules are not available")

        # Check model directory
        if not os.path.exists(self.config.model_path):
            errors.append(f"Model path does not exist: {self.config.model_path}")

        # Check TP configuration
        if hasattr(self.config.infer_args, "parallel"):
            if self.config.infer_args.parallel.tp <= 0:
                errors.append("Tensor parallel size (tp) must be positive")

        return errors

    def _cleanup_framework_resources(self) -> None:
        """InfiniLM-specific resource cleanup"""
        try:
            if hasattr(torch.cuda, "empty_cache") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass

    def unload_model(self) -> None:
        """Unload model"""
        if self.model_instance:
            try:
                if hasattr(self.model_instance, "destroy_model_instance"):
                    self.model_instance.destroy_model_instance()
                logger.info("InfiniLM model unloaded")
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")

            self.model_instance = None

        self.model_loaded = False
        self.tokenizer = None

    def generate(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[List[str], List[float], List[float]]:
        """InfiniLM inference implementation"""
        if not self.model_loaded or not self.model_instance:
            raise RuntimeError("Model not loaded")

        logger.info(f"InfiniLM batch inference for {len(prompts)} prompts")
        logger.info(f"Max tokens: {max_tokens}, Temperature: {temperature}")

        # Encode prompts
        token_lists = []
        for i, prompt in enumerate(prompts):
            tokens = self.tokenizer.encode(prompt)
            token_lists.append(tokens)
            if i == 0:  # Record first prompt information
                logger.debug(f"First prompt: {len(tokens)} tokens")

        tasks = self._create_infer_tasks(token_lists, temperature, top_p, top_k)

        # Execute inference
        return self._execute_batch_inference(tasks, max_tokens)

    def batch_generate(
        self,
        batch_prompts: List[List[str]],
        max_tokens: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[List[List[str]], List[List[float]], List[List[float]]]:
        """
        Batch text generation (multiple batches)

        Note: For large batches, we may need to split to avoid OOM
        """
        logger.info(f"Batch generating for {len(batch_prompts)} batches")

        all_results = []

        for batch_idx, prompts in enumerate(batch_prompts):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batch_prompts)}")

            # Check batch size to avoid OOM
            max_batch_size = 8
            if len(prompts) > max_batch_size:
                logger.warning(f"Batch size {len(prompts)} too large, splitting")
                results = self._split_and_generate(
                    prompts, max_tokens, temperature, top_p, top_k, max_batch_size
                )
            else:
                texts, latencies, ttfts = self.generate(
                    prompts, max_tokens, temperature, top_p, top_k
                )
                results = (texts, latencies, ttfts)

            all_results.append(results)

        return self._reorganize_results(all_results)

    def calculate_perplexity(self, test_data: List[str]) -> float:
        """Calculate perplexity"""
        if not self.model_loaded or not self.model_instance:
            raise RuntimeError("Model not loaded")

        logger.info(f"Calculating perplexity for {len(test_data)} test samples")

        try:
            if hasattr(self.config.infer_args, "max_seq_len"):
                max_seq_len = self.config.infer_args.max_seq_len
            else:
                max_seq_len = DEFAULT_MAX_SEQ_LEN

            # Convert text to token sequences
            test_sequences = [
                self.tokenizer.encode(text)[:max_seq_len] for text in test_data
            ]

            # Use model's perplexity method if available
            if hasattr(self.model_instance, "perplexity"):
                batch_size = min(4, len(test_sequences))
                return self.model_instance.perplexity(
                    test_sequences, batch_size=batch_size
                )
            else:
                logger.warning("Model does not support perplexity calculation")
                return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate perplexity: {e}")
            return 0.0

    def _get_device_type(self):
        """Get device type based on configuration"""
        if DeviceType is None:
            return None

        if hasattr(self.config, "device"):
            if hasattr(self.config.device, "accelerator"):
                accelerator = self.config.device.accelerator.value.lower()
            else:
                accelerator = "nvidia"
        else:
            accelerator = "nvidia"

        if accelerator == "nvidia":
            return DeviceType.DEVICE_TYPE_NVIDIA
        elif accelerator == "cpu":
            return DeviceType.DEVICE_TYPE_CPU
        elif accelerator == "amd":
            return (
                DeviceType.DEVICE_TYPE_AMD
                if hasattr(DeviceType, "DEVICE_TYPE_AMD")
                else DeviceType.DEVICE_TYPE_NVIDIA
            )
        elif accelerator == "intel":
            return (
                DeviceType.DEVICE_TYPE_INTEL
                if hasattr(DeviceType, "DEVICE_TYPE_INTEL")
                else DeviceType.DEVICE_TYPE_NVIDIA
            )
        else:
            logger.warning(f"Unknown accelerator: {accelerator}, using default")
            return DeviceType.DEVICE_TYPE_NVIDIA

    def _create_infer_tasks(self, token_lists, temperature, top_p, top_k):
        """Create inference tasks"""
        tasks = []

        if hasattr(self.config.infer_args, "max_seq_len"):
            config_max_seq_len = self.config.infer_args.max_seq_len
        else:
            config_max_seq_len = DEFAULT_MAX_SEQ_LEN

        for i, tokens in enumerate(token_lists):
            try:
                # End tokens
                if self.eos_token_id:
                    end_tokens = (
                        self.eos_token_id
                        if isinstance(self.eos_token_id, list)
                        else [self.eos_token_id]
                    )
                else:
                    end_tokens = []

                # Maximum sequence length
                max_seq_len = min(
                    config_max_seq_len,
                    (
                        self.model_instance.max_context_len()
                        if hasattr(self.model_instance, "max_context_len")
                        else config_max_seq_len
                    ),
                )

                # Create task
                task = InferTask(
                    id=i,
                    tokens=tokens,
                    max_tokens=max_seq_len,
                    temperature=temperature,
                    topk=top_k,
                    topp=top_p,
                    end_tokens=end_tokens,
                )

                # Bind KV cache
                kv_cache = KVCache(self.model_instance)
                task.bind_kvcache(kv_cache)

                tasks.append((task, kv_cache))

            except Exception as e:
                logger.error(f"Failed to create InferTask {i}: {e}")
                raise

        logger.info(f"Created {len(tasks)} InferTasks")
        return tasks

    def _execute_batch_inference(self, tasks_with_caches, max_tokens):
        """Execute batch inference"""
        tasks = [t[0] for t in tasks_with_caches]
        kv_caches = [t[1] for t in tasks_with_caches]

        generated_texts = []
        latencies = []
        ttfts = []
        all_tokens = [[] for _ in range(len(tasks))]

        try:
            # First inference (TTFT)
            start_time = time.perf_counter()
            output_tokens = self.model_instance.batch_infer_one_round(tasks)
            ttft = (time.perf_counter() - start_time) * 1000

            # Process first output
            for i, output in enumerate(output_tokens):
                if output:
                    token = output[0] if isinstance(output, list) else output
                    all_tokens[i].append(token)
                    ttfts.append(ttft)
                else:
                    all_tokens[i].append(0)
                    ttfts.append(0.0)

            # Continue generating remaining tokens
            generated = 1
            while generated < max_tokens:
                active_tasks = []
                active_indices = []

                for i, (task, tokens_list) in enumerate(zip(tasks, all_tokens)):
                    if not tokens_list:
                        continue

                    last_token = tokens_list[-1]
                    is_eos = self._is_eos_token(last_token)

                    if not is_eos and len(tokens_list) < max_tokens:
                        task.next(last_token)
                        active_tasks.append(task)
                        active_indices.append(i)

                if not active_tasks:
                    break

                # Batch inference
                active_outputs = self.model_instance.batch_infer_one_round(active_tasks)

                # Process outputs
                for idx, task_idx in enumerate(active_indices):
                    if idx < len(active_outputs) and active_outputs[idx]:
                        token = (
                            active_outputs[idx][0]
                            if isinstance(active_outputs[idx], list)
                            else active_outputs[idx]
                        )
                        all_tokens[task_idx].append(token)

                generated += 1

                if generated % 10 == 0:
                    logger.debug(f"Generated {generated}/{max_tokens} tokens")

            # Compute total latency and decode text
            total_latency = (time.perf_counter() - start_time) * 1000

            for i, tokens in enumerate(all_tokens):
                latencies.append(total_latency)

                if tokens:
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    generated_texts.append(text)
                else:
                    generated_texts.append("")

        finally:
            # Clean up KV caches
            self._cleanup_kv_caches(kv_caches)

        logger.info(f"Inference completed: {len(generated_texts)} prompts")
        return generated_texts, latencies, ttfts

    def _is_eos_token(self, token):
        """Check whether token is EOS"""
        if not self.eos_token_id:
            return False

        if isinstance(self.eos_token_id, list):
            return token in self.eos_token_id
        else:
            return token == self.eos_token_id

    def _cleanup_kv_caches(self, kv_caches):
        """Clean up KV caches"""
        logger.info("Cleaning up KVCaches")
        for i, kv_cache in enumerate(kv_caches):
            try:
                if kv_cache and self.model_instance:
                    kv_cache.drop(self.model_instance)
            except Exception as e:
                logger.warning(f"Failed to drop KV cache {i}: {e}")

    def _split_and_generate(
        self, prompts, max_tokens, temperature, top_p, top_k, max_batch_size
    ):
        """Split and generate"""
        texts, latencies, ttfts = [], [], []

        for i in range(0, len(prompts), max_batch_size):
            sub_prompts = prompts[i : i + max_batch_size]
            logger.debug(f"Processing sub-batch {i // max_batch_size + 1}")

            sub_texts, sub_latencies, sub_ttfts = self.generate(
                sub_prompts, max_tokens, temperature, top_p, top_k
            )

            texts.extend(sub_texts)
            latencies.extend(sub_latencies)
            ttfts.extend(sub_ttfts)

        return texts, latencies, ttfts

    def _reorganize_results(self, all_results):
        """Reorganize results"""
        all_texts, all_latencies, all_ttfts = [], [], []

        for texts, latencies, ttfts in all_results:
            all_texts.append(texts)
            all_latencies.append(latencies)
            all_ttfts.append(ttfts)

        return all_texts, all_latencies, all_ttfts

    def get_peak_memory_usage(self) -> Optional[float]:
        """Get peak accelerator memory usage (GB) via AcceleratorMonitor when possible."""
        try:
            dev = getattr(self.config, "device", {}) or {}
            if isinstance(dev, dict):
                accelerator = dev.get("accelerator", "nvidia")
                device_ids = dev.get("device_ids")
            else:
                accelerator = getattr(dev, "accelerator", "nvidia")
                device_ids = getattr(dev, "device_ids", None)

            from infinibench.utils.accelerator_monitor import (
                create_accelerator_monitor,
            )

            mon = create_accelerator_monitor(
                accelerator_type=accelerator, device_ids=device_ids
            )

            # Use framework API first
            peak_bytes = mon.get_peak_memory_allocated()
            if peak_bytes is not None and peak_bytes > 0:
                return peak_bytes / (1024**3)

            # fallback
            peak_gb = mon.get_peak_memory_gb()
            return peak_gb if peak_gb > 0 else None

        except Exception as e:
            logger.debug(f"Failed to get peak memory via monitor: {e}")
            return None

    def _validate_framework_config(self) -> List[str]:
        """Validate framework-specific configuration"""
        return self.validate_config()

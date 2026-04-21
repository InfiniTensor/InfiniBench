#!/usr/bin/env python3
import vllm
import logging, os, time
from typing import List, Tuple, Dict, Any
from vllm import LLM, SamplingParams

from infinibench.common.constants import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_MAX_SEQ_LEN,
)

logger = logging.getLogger(__name__)


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_nested(obj, path, default=None):
    cur = obj
    for p in path.split("."):
        cur = _get(cur, p, None)
        if cur is None:
            return default
    return cur


class VLLMImpl:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.vllm_version = getattr(vllm, "__version__", "unknown")
        logger.info(f"vLLM version: {self.vllm_version}")

    def _setup_device_visibility(self):
        dev = getattr(self.config, "device", {})
        cpu_only = bool(_get(dev, "cpu_only", False))
        if cpu_only:
            logger.info("vLLM CPU-only mode")
            return

        accelerator = _get_nested(dev, "accelerator.value", None)
        if accelerator is None:
            accelerator = _get(dev, "accelerator", "nvidia")
        accelerator = str(accelerator).lower()

        device_ids = _get(dev, "device_ids", None)

        # Only set when accelerator is NVIDIA and device_ids are explicitly specified
        if accelerator == "nvidia" and device_ids:
            device_str = ",".join(str(d) for d in device_ids)
            if not os.environ.get("CUDA_VISIBLE_DEVICES"):
                os.environ["CUDA_VISIBLE_DEVICES"] = device_str
                logger.info(f"Set CUDA_VISIBLE_DEVICES={device_str}")

    def load_model(self) -> None:
        logger.info(f"Loading vLLM model from {self.config.model_path}")
        self._setup_device_visibility()

        ia = getattr(self.config, "infer_args", {}) or {}
        tp = int(_get_nested(ia, "parallel.tp", 1) or 1)
        max_seq_len = int(
            _get(ia, "max_seq_len", DEFAULT_MAX_SEQ_LEN) or DEFAULT_MAX_SEQ_LEN
        )

        framework_kwargs = _get(ia, "framework_kwargs", {}) or {}
        if not isinstance(framework_kwargs, dict):
            framework_kwargs = {}

        # vLLM LLM kwargs
        vllm_kwargs: Dict[str, Any] = {
            "model": str(self.config.model_path),
            "tensor_parallel_size": tp,
            "max_model_len": max_seq_len,
        }

        # Common optional arguments
        # dtype / trust_remote_code / seed / gpu_memory_utilization / swap_space etc.
        # are all allowed to be overridden via framework_kwargs
        allow = {
            "dtype",
            "trust_remote_code",
            "seed",
            "gpu_memory_utilization",
            "swap_space",
            "disable_log_stats",
            "enforce_eager",
            "quantization",
            "max_num_batched_tokens",
            "max_num_seqs",
            "block_size",
        }
        for k, v in framework_kwargs.items():
            if k in allow and v is not None:
                vllm_kwargs[k] = v

        self.model = LLM(**vllm_kwargs)
        self.model_loaded = True
        self.tokenizer = self.model.get_tokenizer()
        logger.info("vLLM model loaded")

    def unload_model(self) -> None:
        # vLLM does not provide an explicit unload; release references as much as possible
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("vLLM model unloaded")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[List[str], List[float], List[float]]:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        if not prompts:
            return [], [], []

        sampling_params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k) if int(top_k) > 0 else -1,
        )

        texts, lat_ms, ttft_ms = [], [], []

        for i, prompt in enumerate(prompts):
            try:
                start = time.perf_counter()
                outputs = self.model.generate([prompt], sampling_params)
                out = outputs[0]
                latency = (time.perf_counter() - start) * 1000.0
                lat_ms.append(latency)

                # Compatible with metrics fields across different vLLM versions
                ttft = None
                m = getattr(out, "metrics", None)
                if m is not None:
                    ft = getattr(m, "first_token_time", None)
                    if ft is not None:
                        ttft = float(ft) * 1000.0
                if ttft is None:
                    ttft = latency * 0.1
                ttft_ms.append(ttft)

                # debug text
                if i < 3:
                    if hasattr(out, "outputs") and out.outputs:
                        texts.append(out.outputs[0].text)
                    else:
                        texts.append(str(out))
            except Exception as e:
                logger.error(f"vLLM generation failed for sample {i}: {e}")
                lat_ms.append(0.0)
                ttft_ms.append(0.0)

        return texts, lat_ms, ttft_ms

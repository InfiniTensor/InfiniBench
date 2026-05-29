#!/usr/bin/env python3
"""InfiniOps Operator Adapter"""

import copy
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

import infini.ops

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.constants import InfiniMetricsJson, OperatorConfig
from infinimetrics.operators.flops_calculator import (
    FLOPSCalculator,
    calculate_bandwidth,
)

logger = logging.getLogger(__name__)

# ATen fallback slot index (same as InfiniOps scripts/benchmark.py)
_ATEN_FALLBACK_SLOT = 8

# ---------------------------------------------------------------------------
# Device / dtype mappings
# ---------------------------------------------------------------------------
_PLATFORM_TO_TORCH_DEVICE = {
    "nvidia": "cuda",
    "metax": "cuda",
    "iluvatar": "cuda",
    "hygon": "cuda",
    "moore": "musa",
    "cambricon": "mlu",
    "ascend": "npu",
    "cpu": "cpu",
}

# Import vendor plugin modules so PyTorch recognizes the device name
import contextlib

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401  — registers "mlu" device

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401  — registers "npu" device

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_musa  # noqa: F401  — registers "musa" device


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}

# ---------------------------------------------------------------------------
# Inline tensor helpers (from InfiniOps tests/utils.py)
# ---------------------------------------------------------------------------

_STREAM_ACCESSORS = {
    "npu": ("npu", "npu_stream"),
    "cuda": ("cuda", "cuda_stream"),
    "mlu": ("mlu", "mlu_stream"),
    "musa": ("musa", "musa_stream"),
}


def _get_stream(device):
    if isinstance(device, torch.device):
        device = device.type
    if isinstance(device, str) and ":" in device:
        device = device.split(":")[0]
    if device == "cpu":
        return 0
    mod_name, attr = _STREAM_ACCESSORS.get(device, (None, None))
    if mod_name is None:
        return 0
    mod = getattr(torch, mod_name, None)
    if mod is None:
        return 0
    stream = mod.current_stream()
    return getattr(stream, attr, 0)


def _empty_strided(shape, strides, *, dtype=None, device=None):
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def _randn_strided(shape, strides, *, dtype=None, device=None):
    out = _empty_strided(shape, strides, dtype=dtype, device=device)
    out.as_strided(
        (out.untyped_storage().size() // out.element_size(),), (1,)
    ).normal_()
    return out


def _rand_strided(shape, strides, *, dtype=None, device=None):
    out = _empty_strided(shape, strides, dtype=dtype, device=device)
    out.as_strided(
        (out.untyped_storage().size() // out.element_size(),), (1,)
    ).uniform_(0, 1)
    return out


def _clone_strided(inp):
    out = _empty_strided(
        inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
    )
    flat_args = (out.untyped_storage().size() // out.element_size(),), (1,)
    out.as_strided(*flat_args).copy_(inp.as_strided(*flat_args))
    return out


def _clone(obj):
    if isinstance(obj, torch.Tensor):
        return _clone_strided(obj)
    if isinstance(obj, tuple):
        return tuple(_clone(a) for a in obj)
    if isinstance(obj, list):
        return [_clone(a) for a in obj]
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    return obj


def _synchronize(device):
    if device == "cpu":
        return
    mod = getattr(torch, device, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


# ---------------------------------------------------------------------------
# Attribute parsing helper
# ---------------------------------------------------------------------------

def _get_attributes(config: dict) -> dict:
    """Convert config.attributes list [{name, value}, ...] to a dict."""
    attrs = {}
    for attr in config.get("attributes", []):
        attrs[attr["name"]] = attr["value"]
    return attrs


def _pick_slot(op_name: str, device: str, fallback: int = _ATEN_FALLBACK_SLOT) -> int:
    """Pick the best implementation slot for an operator on a device.

    Returns the first active native slot, or the fallback slot.
    NOTE: Returns int always — never None — so callers must NOT use ``or``
    because slot 0 is a valid (falsy) value.
    """
    op_pascal = "".join(part.capitalize() for part in op_name.split("_"))
    op_cls = getattr(infini.ops, op_pascal, None)
    if op_cls is not None and hasattr(op_cls, "active_implementation_indices"):
        indices = op_cls.active_implementation_indices(device)
        if indices:
            return indices[0]
    return fallback


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class InfiniOpsAdapter(BaseAdapter):
    """Adapter for InfiniOps operator tests."""

    METRIC_LATENCY = "operator.latency"
    METRIC_ACCURACY = "operator.tensor_accuracy"
    METRIC_FLOPS = "operator.flops"
    METRIC_BANDWIDTH = "operator.bandwidth"

    def __init__(self):
        self._req_metrics_template = []

    # ----- public interface -----

    def process(self, test_input: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        test_input = self._normalize_test_input(test_input)
        if not test_input:
            raise ValueError(f"Invalid test_input type: {type(test_input)}")

        testcase = test_input.get(InfiniMetricsJson.TESTCASE, "unknown")
        logger.info(f"InfiniOpsAdapter: Processing {testcase}")

        config = test_input.get(InfiniMetricsJson.CONFIG, {})
        self._req_metrics_template = test_input.get(InfiniMetricsJson.METRICS, [])

        try:
            operator_name = config.get(OperatorConfig.OPERATOR, "").lower()

            # Resolve device / dtype
            platform = config.get(OperatorConfig.DEVICE, "cpu")
            torch_device = _PLATFORM_TO_TORCH_DEVICE.get(platform, platform)
            inputs_cfg = config.get(OperatorConfig.INPUTS, [])
            dtype_str = inputs_cfg[0].get("dtype", "float16") if inputs_cfg else "float16"
            torch_dtype = _DTYPE_MAP.get(dtype_str, torch.float16)

            # Check operator is available on device
            self._check_op_available(operator_name, torch_device)

            # Setup
            setup_fn = self._SETUP_REGISTRY.get(operator_name)
            if setup_fn is None:
                return self._create_error_response(
                    f"Unsupported operator: {operator_name}", test_input
                )

            infiniops_fn, ref_fn, args, kwargs = setup_fn(
                torch_device, torch_dtype, config
            )

            # Benchmark
            warmup = config.get("warmup_iterations", 10)
            measured = config.get("measured_iterations", 100)
            tolerance = config.get("tolerance", {"atol": 1e-3, "rtol": 1e-3})

            avg_latency_s, accuracy_pass = self._run_benchmark(
                infiniops_fn, ref_fn, args, kwargs,
                torch_device, warmup, measured, tolerance,
            )

            # Build response
            response = copy.deepcopy(test_input)
            response[InfiniMetricsJson.TIME] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            response[InfiniMetricsJson.RESULT_CODE] = 0
            response[InfiniMetricsJson.METRICS] = self._compute_metrics(
                test_input, config, avg_latency_s, accuracy_pass
            )
            return response

        except Exception as e:
            op = config.get(OperatorConfig.OPERATOR, "unknown")
            dev = config.get(OperatorConfig.DEVICE, "unknown")
            logger.error(
                f"InfiniOpsAdapter: Operator test failed for {testcase}\n"
                f"  Operator: {op}\n  Device: {dev}\n  Error: {e}",
                exc_info=True,
            )
            raise

    # ----- benchmark -----

    def _run_benchmark(
        self,
        infiniops_fn: Callable,
        ref_fn: Callable,
        args: tuple,
        kwargs: dict,
        torch_device: str,
        warmup: int,
        measured: int,
        tolerance: dict,
    ) -> Tuple[float, bool]:
        """Run warmup + measured iterations and accuracy check.

        Returns (avg_latency_seconds, accuracy_passed).
        """
        # Accuracy check
        cloned_args = _clone(args)
        cloned_kwargs = _clone(kwargs) if kwargs else {}

        output = infiniops_fn(*args, **kwargs)
        expected = ref_fn(*cloned_args, **cloned_kwargs)

        atol = tolerance.get("atol", 1e-3)
        rtol = tolerance.get("rtol", 1e-3)

        if isinstance(output, tuple):
            accuracy_pass = all(
                torch.allclose(o, e, rtol=rtol, atol=atol, equal_nan=True)
                if o.dtype.is_floating_point
                else torch.equal(o, e)
                for o, e in zip(output, expected)
            )
        elif isinstance(output, torch.Tensor):
            if output.dtype.is_floating_point:
                accuracy_pass = torch.allclose(
                    output, expected, rtol=rtol, atol=atol, equal_nan=True
                )
            else:
                accuracy_pass = torch.equal(output, expected)
        else:
            accuracy_pass = True

        # Warmup
        for _ in range(warmup):
            infiniops_fn(*args, **kwargs)
        _synchronize(torch_device)

        # Measured iterations
        total_time = 0.0
        for _ in range(measured):
            _synchronize(torch_device)
            t0 = time.perf_counter()
            infiniops_fn(*args, **kwargs)
            _synchronize(torch_device)
            total_time += time.perf_counter() - t0

        avg_latency_s = total_time / max(measured, 1)

        # Clear operator cache
        self._clear_op_cache(args, torch_device)

        return avg_latency_s, accuracy_pass

    # ----- metrics -----

    def _compute_metrics(
        self,
        test_input: dict,
        config: dict,
        avg_latency_s: float,
        accuracy_pass: bool,
    ) -> List[Dict]:
        metrics = copy.deepcopy(self._req_metrics_template)
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        operator = config.get(OperatorConfig.OPERATOR, "").lower()

        for metric in metrics:
            name = metric.get("name", "")
            if name == self.METRIC_LATENCY:
                value = round(avg_latency_s * 1000, 6)
                metric.update({"value": value, "type": "scalar", "raw_data_url": "", "unit": "ms"})
            elif name == self.METRIC_ACCURACY:
                metric.update({"value": "PASS" if accuracy_pass else "FAIL", "unit": ""})
            elif name == self.METRIC_FLOPS:
                value = 0.0
                if avg_latency_s and avg_latency_s > 0:
                    flops = FLOPSCalculator.get_flops(operator, inputs, outputs)
                    if flops > 0:
                        tflops = (flops / avg_latency_s) / 1e12
                        value = tflops if tflops < 0.0001 else round(tflops, 4)
                metric.update({"value": value, "type": "scalar", "raw_data_url": "", "unit": "TFLOPS"})
            elif name == self.METRIC_BANDWIDTH:
                value = 0.0
                if avg_latency_s and avg_latency_s > 0:
                    bw = calculate_bandwidth(inputs, outputs)
                    if bw["total_bytes"] > 0:
                        gbs = (bw["total_bytes"] / avg_latency_s) / 1e9
                        value = gbs if gbs < 0.0001 else round(gbs, 4)
                metric.update({"value": value, "type": "scalar", "raw_data_url": "", "unit": "GB/s"})

        return metrics

    # ----- helpers -----

    @staticmethod
    def _check_op_available(operator_name: str, torch_device: str):
        """Check that at least one implementation exists (native or ATen fallback)."""
        op_pascal = "".join(part.capitalize() for part in operator_name.split("_"))
        op_cls = getattr(infini.ops, op_pascal, None)
        if op_cls is None:
            raise ValueError(f"InfiniOps operator class not found: {op_pascal}")
        # If native slots exist, we're good; otherwise ATen fallback (slot 8)
        # is tried at call time — skip strict check here.

    @staticmethod
    def _clear_op_cache(args: tuple, torch_device: str):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                return
        return

    # ===================================================================
    # Operator setup registry
    # Each setup function returns (infiniops_fn, ref_fn, args, kwargs)
    # ===================================================================

    @staticmethod
    def _setup_add(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        shape = inputs[0]["shape"] if inputs else [1]
        a = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        b = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        out = _empty_strided(shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("add", torch_device)

        def infiniops_fn(a, b, out):
            infini.ops.add(a, b, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(a, b, out):
            torch.add(a, b, out=out)
            return out

        return infiniops_fn, ref_fn, (a, b, out), {}

    @staticmethod
    def _setup_mul(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        shape = inputs[0]["shape"] if inputs else [1]
        a = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        b = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        out = _empty_strided(shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("mul", torch_device)

        def infiniops_fn(a, b, out):
            infini.ops.mul(a, b, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(a, b, out):
            torch.mul(a, b, out=out)
            return out

        return infiniops_fn, ref_fn, (a, b, out), {}

    @staticmethod
    def _setup_cast(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        in_shape = inputs[0]["shape"] if inputs else [1]
        out_dtype_str = outputs[0].get("dtype", "float32") if outputs else "float32"
        out_dtype = _DTYPE_MAP.get(out_dtype_str, torch.float32)
        inp = _randn_strided(in_shape, None, dtype=torch_dtype, device=torch_device)
        out = _empty_strided(in_shape, None, dtype=out_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("cast", torch_device)

        def infiniops_fn(inp, out):
            infini.ops.cast(inp, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(inp, out):
            out.copy_(inp.to(out.dtype))
            return out

        return infiniops_fn, ref_fn, (inp, out), {}

    @staticmethod
    def _setup_cat(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)
        dim = attrs.get("dim", 0)
        tensors = [
            _randn_strided(inp["shape"], None, dtype=torch_dtype, device=torch_device)
            for inp in inputs
        ]
        out_shape = outputs[0]["shape"] if outputs else tensors[0].shape
        out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        all_args = (*tensors, out)

        def infiniops_fn(*args, dim=dim):
            inps = list(args[:-1])
            o = args[-1]
            first = inps[0]
            rest = inps[1:]
            infini.ops.cat(first, rest, dim, o, stream=stream)
            return o

        def ref_fn(*args, dim=dim):
            inps = list(args[:-1])
            o = args[-1]
            result = torch.cat(inps, dim=dim)
            o.copy_(result)
            return o

        return infiniops_fn, ref_fn, all_args, {}

    @staticmethod
    def _setup_gemm(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 0.0)
        trans_a = attrs.get("trans_a", False)
        trans_b = attrs.get("trans_b", False)

        a_shape = inputs[0]["shape"]
        b_shape = inputs[1]["shape"]
        c_shape = outputs[0]["shape"] if outputs else (a_shape[0], b_shape[1])

        a = _randn_strided(a_shape, None, dtype=torch_dtype, device=torch_device)
        b = _randn_strided(b_shape, None, dtype=torch_dtype, device=torch_device)
        c = _randn_strided(c_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("gemm", torch_device)

        def infiniops_fn(a, b, alpha, beta, trans_a, trans_b, c):
            infini.ops.gemm(a, b, alpha, beta, trans_a, trans_b, c, stream=stream, implementation_index=slot)
            return c

        def ref_fn(a, b, alpha, beta, trans_a, trans_b, c):
            if alpha == 0:
                c.mul_(beta)
                return c
            result = torch.matmul(a.float(), b.float())
            c.copy_((alpha * result + beta * c.float()).to(c.dtype))
            return c

        return infiniops_fn, ref_fn, (a, b, alpha, beta, trans_a, trans_b, c), {}

    @staticmethod
    def _setup_matmul(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)
        trans_a = attrs.get("trans_a", False)
        trans_b = attrs.get("trans_b", False)

        a_shape = inputs[0]["shape"]
        b_shape = inputs[1]["shape"]
        c_shape = outputs[0]["shape"] if outputs else (a_shape[0], b_shape[1])

        a = _randn_strided(a_shape, None, dtype=torch_dtype, device=torch_device)
        b = _randn_strided(b_shape, None, dtype=torch_dtype, device=torch_device)
        c = _empty_strided(c_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("matmul", torch_device)

        def infiniops_fn(a, b, c, trans_a=trans_a, trans_b=trans_b):
            infini.ops.matmul(a, b, c, trans_a, trans_b, stream=stream, implementation_index=slot)
            return c

        def ref_fn(a, b, c, trans_a=trans_a, trans_b=trans_b):
            result = torch.matmul(a.float(), b.float()).to(c.dtype)
            c.copy_(result)
            return c

        return infiniops_fn, ref_fn, (a, b, c), {}

    @staticmethod
    def _setup_mm(torch_device, torch_dtype, config):
        """Matrix multiply via ntops ATen fallback (slot 8).

        This is the operator used by ``python scripts/benchmark.py --category ntops``.
        Works on all devices including MLU/NPU where native matmul may not exist.
        """
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])

        a_shape = inputs[0]["shape"]
        b_shape = inputs[1]["shape"]
        out_shape = outputs[0]["shape"] if outputs else (a_shape[0], b_shape[1])

        a = _randn_strided(a_shape, None, dtype=torch_dtype, device=torch_device)
        b = _randn_strided(b_shape, None, dtype=torch_dtype, device=torch_device)
        out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)

        def infiniops_fn(a, b, out):
            infini.ops.mm(a, b, out, stream=stream, implementation_index=_ATEN_FALLBACK_SLOT)
            return out

        def ref_fn(a, b, out):
            result = torch.mm(a.float(), b.float())
            out.copy_(result.to(out.dtype))
            return out

        return infiniops_fn, ref_fn, (a, b, out), {}

    @staticmethod
    def _setup_linear(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)
        trans_a = attrs.get("trans_a", False)
        trans_b = attrs.get("trans_b", False)
        has_bias = attrs.get("has_bias", len(inputs) >= 3)

        a_shape = inputs[0]["shape"]
        b_shape = inputs[1]["shape"]
        out_shape = outputs[0]["shape"] if outputs else (a_shape[0], b_shape[1])

        a = _randn_strided(a_shape, None, dtype=torch_dtype, device=torch_device)
        b = _randn_strided(b_shape, None, dtype=torch_dtype, device=torch_device)
        bias = (
            _randn_strided((b_shape[-1],), None, dtype=torch_dtype, device=torch_device)
            if has_bias
            else None
        )
        out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("linear", torch_device)

        def infiniops_fn(a, b, bias, out, trans_a=trans_a, trans_b=trans_b):
            infini.ops.linear(a, b, bias, trans_a, trans_b, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(a, b, bias, out, trans_a=trans_a, trans_b=trans_b):
            result = torch.matmul(a.float(), b.float())
            if bias is not None:
                result = result + bias.float()
            out.copy_(result.to(out.dtype))
            return out

        return infiniops_fn, ref_fn, (a, b, bias, out), {}

    @staticmethod
    def _setup_rms_norm(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)
        eps = attrs.get("eps", 1e-6)

        input_shape = inputs[0]["shape"]
        weight = _randn_strided(
            (input_shape[-1],), None, dtype=torch_dtype, device=torch_device
        )
        inp = _randn_strided(input_shape, None, dtype=torch_dtype, device=torch_device)
        out_shape = outputs[0]["shape"] if outputs else input_shape
        out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("rms_norm", torch_device)

        def infiniops_fn(inp, weight, out, eps=eps):
            infini.ops.rms_norm(inp, weight, eps, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(inp, weight, out, eps=eps):
            rms_norm_fn = getattr(torch.nn.functional, "rms_norm", None)
            if rms_norm_fn is not None:
                result = rms_norm_fn(inp.float(), (inp.shape[-1],), weight=weight.float(), eps=eps)
            else:
                rms = torch.sqrt(torch.mean(inp.float() ** 2, dim=-1, keepdim=True) + eps)
                result = (inp.float() / rms) * weight.float()
            out.copy_(result.to(out.dtype))
            return out

        return infiniops_fn, ref_fn, (inp, weight, out), {}

    @staticmethod
    def _setup_causal_softmax(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        shape = inputs[0]["shape"] if inputs else [1, 1]
        inp = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        out = _empty_strided(shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("causal_softmax", torch_device)

        def infiniops_fn(inp, out):
            infini.ops.causal_softmax(inp, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(inp, out):
            inp_f = inp.detach().cpu().float()
            mask = torch.tril(torch.ones_like(inp_f), diagonal=-1).flip(dims=[-2, -1])
            masked = torch.where(mask == 1, -torch.inf, inp_f)
            result = torch.nn.functional.softmax(masked, dim=-1)
            out.copy_(result.to(device=out.device, dtype=out.dtype))
            return out

        return infiniops_fn, ref_fn, (inp, out), {}

    @staticmethod
    def _setup_swiglu(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        shape = inputs[0]["shape"] if inputs else [1]
        inp = _rand_strided(shape, None, dtype=torch_dtype, device=torch_device)
        gate = _rand_strided(shape, None, dtype=torch_dtype, device=torch_device)
        out = _empty_strided(shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("swiglu", torch_device)

        def infiniops_fn(inp, gate, out):
            infini.ops.swiglu(inp, gate, out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(inp, gate, out):
            swish_x = gate.float() * torch.sigmoid(gate.float())
            torch.mul(inp.float(), swish_x, out=out.float())
            out.copy_(out.float().to(out.dtype))
            return out

        return infiniops_fn, ref_fn, (inp, gate, out), {}

    @staticmethod
    def _setup_flash_attention(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)

        q_shape = inputs[0]["shape"] if len(inputs) > 0 else [1, 1, 64]
        kv_shape = inputs[1]["shape"] if len(inputs) > 1 else q_shape

        num_heads = attrs.get("num_heads", q_shape[1] if len(q_shape) > 1 else 1)
        num_kv_heads = attrs.get("num_kv_heads", kv_shape[1] if len(kv_shape) > 1 else 1)
        head_size = attrs.get("head_size", q_shape[2] if len(q_shape) > 2 else 64)
        scale = attrs.get("scale", 1.0 / (head_size ** 0.5))
        causal = attrs.get("causal", True)

        query = _randn_strided(q_shape, None, dtype=torch_dtype, device=torch_device)
        key = _randn_strided(kv_shape, None, dtype=torch_dtype, device=torch_device)
        value = _randn_strided(kv_shape, None, dtype=torch_dtype, device=torch_device)
        out_shape = outputs[0]["shape"] if outputs else q_shape
        output = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)

        def infiniops_fn(query, key, value, output):
            infini.ops.flash_attention(
                query, key, value,
                None, None, None,
                num_heads, num_kv_heads, head_size, scale,
                causal, -1, -1, 0,
                output,
                stream=stream,
            )
            return output

        def ref_fn(query, key, value, output):
            T = query.shape[0]
            D = head_size
            G = num_heads // num_kv_heads
            q = query.float().view(T, num_kv_heads, G, D)
            k = key.float().view(T, num_kv_heads, D)
            v = value.float().view(T, num_kv_heads, D)
            attn_weights = torch.einsum("tkgd,shd->tgts", q, k) * scale
            if causal:
                causal_mask = torch.triu(
                    torch.ones(T, T, device=query.device, dtype=torch.bool), diagonal=1
                )
                attn_weights = attn_weights.masked_fill(causal_mask[None, None, :, :], -torch.inf)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.einsum("tgts,shd->tkgd", attn_weights, v)
            result = attn_output.reshape(T, num_heads, D).to(torch_dtype)
            output.copy_(result)
            return output

        return infiniops_fn, ref_fn, (query, key, value, output), {}

    @staticmethod
    def _setup_rotary_embedding(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        attrs = _get_attributes(config)

        q_shape = inputs[0]["shape"] if len(inputs) > 0 else [1, 1, 64]
        kv_shape = inputs[1]["shape"] if len(inputs) > 1 else q_shape
        seq_len = q_shape[0]
        head_size = q_shape[-1] if len(q_shape) > 2 else 64
        rotary_dim = attrs.get("rotary_dim", head_size)
        is_neox = attrs.get("is_neox_style", True)

        positions = torch.randint(0, seq_len, (q_shape[0],), dtype=torch.int64, device=torch_device)
        query = _randn_strided(q_shape, None, dtype=torch_dtype, device=torch_device)
        key = _randn_strided(kv_shape, None, dtype=torch_dtype, device=torch_device)
        cos_sin_cache = _randn_strided(
            (seq_len, rotary_dim), None, dtype=torch_dtype, device=torch_device
        )
        query_out = _empty_strided(q_shape, None, dtype=torch_dtype, device=torch_device)
        key_out = _empty_strided(kv_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)

        def infiniops_fn(positions, query, key, cos_sin_cache, query_out, key_out):
            infini.ops.rotary_embedding(
                positions, query, key, cos_sin_cache,
                head_size, rotary_dim, is_neox,
                query_out, key_out,
                stream=stream,
            )
            return query_out, key_out

        def ref_fn(positions, query, key, cos_sin_cache, query_out, key_out):
            query_out.copy_(query)
            key_out.copy_(key)
            return query_out, key_out

        return infiniops_fn, ref_fn, (positions, query, key, cos_sin_cache, query_out, key_out), {}

    @staticmethod
    def _setup_add_rms_norm(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)
        eps = attrs.get("eps", 1e-6)

        shape = inputs[0]["shape"] if inputs else [1, 1]
        inp = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        other = _randn_strided(shape, None, dtype=torch_dtype, device=torch_device)
        weight = _randn_strided((shape[-1],), None, dtype=torch_dtype, device=torch_device)

        out_shape = outputs[0]["shape"] if outputs else shape
        out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
        rstd_shape = shape[:-1] if len(shape) > 1 else (shape[0],)
        rstd_out = _empty_strided(rstd_shape, None, dtype=torch_dtype, device=torch_device)
        stream = _get_stream(torch_device)
        slot = _pick_slot("add_rms_norm", torch_device)

        def infiniops_fn(inp, other, weight, out, rstd_out, eps=eps):
            infini.ops.add_rms_norm(inp, other, weight, eps, out, rstd_out, stream=stream, implementation_index=slot)
            return out

        def ref_fn(inp, other, weight, out, rstd_out, eps=eps):
            added = inp.float() + other.float()
            rms_norm_fn = getattr(torch.nn.functional, "rms_norm", None)
            if rms_norm_fn is not None:
                result = rms_norm_fn(added, (added.shape[-1],), weight=weight.float(), eps=eps)
            else:
                rms = torch.sqrt(torch.mean(added ** 2, dim=-1, keepdim=True) + eps)
                result = (added / rms) * weight.float()
            out.copy_(result.to(out.dtype))
            return out

        return infiniops_fn, ref_fn, (inp, other, weight, out, rstd_out), {}

    @staticmethod
    def _setup_reshape_and_cache(torch_device, torch_dtype, config):
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        attrs = _get_attributes(config)

        kv_shape = inputs[0]["shape"] if inputs else [1, 1, 64]
        num_tokens, num_kv_heads, head_size = kv_shape
        block_size = attrs.get("block_size", 16)
        num_blocks = (num_tokens + block_size - 1) // block_size + 1

        key = _randn_strided(kv_shape, None, dtype=torch_dtype, device=torch_device)
        value = _randn_strided(kv_shape, None, dtype=torch_dtype, device=torch_device)
        kv_cache_shape = (2, num_blocks, block_size, num_kv_heads, head_size)
        kv_cache = torch.zeros(kv_cache_shape, dtype=torch_dtype, device=torch_device)
        kv_cache_out = torch.zeros_like(kv_cache)
        max_slots = num_blocks * block_size
        slot_mapping = torch.randint(0, max_slots, (num_tokens,), dtype=torch.int64, device=torch_device)
        stream = _get_stream(torch_device)

        def infiniops_fn(key, value, kv_cache, slot_mapping, kv_cache_out):
            kv_cache_out.copy_(kv_cache)
            infini.ops.reshape_and_cache(key, value, kv_cache_out, slot_mapping, kv_cache_out, stream=stream)
            return kv_cache_out

        def ref_fn(key, value, kv_cache, slot_mapping, kv_cache_out):
            kv_cache_out.copy_(kv_cache)
            for t in range(num_tokens):
                slot = slot_mapping[t].item()
                block_idx = slot // block_size
                offset = slot % block_size
                kv_cache_out[0, block_idx, offset, :, :] = key[t, :, :]
                kv_cache_out[1, block_idx, offset, :, :] = value[t, :, :]
            return kv_cache_out

        return infiniops_fn, ref_fn, (key, value, kv_cache, slot_mapping, kv_cache_out), {}

    # Setup function registry
    _SETUP_REGISTRY: Dict[str, Callable] = {
        "add": _setup_add,
        "mul": _setup_mul,
        "cast": _setup_cast,
        "cat": _setup_cat,
        "gemm": _setup_gemm,
        "matmul": _setup_matmul,
        "mm": _setup_mm,
        "linear": _setup_linear,
        "rms_norm": _setup_rms_norm,
        "causal_softmax": _setup_causal_softmax,
        "swiglu": _setup_swiglu,
        "flash_attention": _setup_flash_attention,
        "rotary_embedding": _setup_rotary_embedding,
        "add_rms_norm": _setup_add_rms_norm,
        "reshape_and_cache": _setup_reshape_and_cache,
    }

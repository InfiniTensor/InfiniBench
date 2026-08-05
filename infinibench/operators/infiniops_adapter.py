#!/usr/bin/env python3
"""InfiniOps operator performance adapter."""

import copy
import importlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

import infini.ops

from infinibench.adapter import BaseAdapter
from infinibench.common.constants import (
    AttributeSpec,
    BandwidthField,
    DEFAULT_MEASURED_ITERATIONS,
    DEFAULT_TOLERANCE,
    DEFAULT_WARMUP_ITERATIONS,
    ErrorCode,
    INFINIOPS_DEVICE_PLUGIN_MODULES,
    INFINIOPS_PLATFORM_TO_TORCH_DEVICE,
    INFINIOPS_STREAM_ACCESSORS,
    InfiniBenchJson,
    MetricSpec,
    MetricType,
    OperatorConfig,
    OperatorMetric,
    TensorSpec,
)
from infinibench.operators.flops_calculator import (
    FLOPSCalculator,
    calculate_bandwidth,
)

logger = logging.getLogger(__name__)


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}


class OperatorFamily:
    """Builder families used by the declarative operator registry."""

    BINARY = "binary"
    CAST = "cast"
    CONCAT = "concat"
    MATRIX = "matrix"


@dataclass(frozen=True)
class OperatorSpec:
    """Description of a common operator and its builder family."""

    name: str
    family: str
    torch_name: Optional[str] = None
    scalar_args: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class BenchmarkCase:
    """Prepared InfiniOps and reference calls for one benchmark."""

    operation: Callable
    reference: Callable
    args: tuple
    kwargs: dict = field(default_factory=dict)


OPERATOR_SPECS: Dict[str, OperatorSpec] = {
    "add": OperatorSpec("add", OperatorFamily.BINARY, torch_name="add"),
    "sub": OperatorSpec(
        "sub", OperatorFamily.BINARY, torch_name="sub", scalar_args=(1.0,)
    ),
    "mul": OperatorSpec("mul", OperatorFamily.BINARY, torch_name="mul"),
    "div": OperatorSpec("div", OperatorFamily.BINARY, torch_name="div"),
    "cast": OperatorSpec("cast", OperatorFamily.CAST),
    "cat": OperatorSpec("cat", OperatorFamily.CONCAT),
    "gemm": OperatorSpec("gemm", OperatorFamily.MATRIX),
    "matmul": OperatorSpec("matmul", OperatorFamily.MATRIX),
    "mm": OperatorSpec("mm", OperatorFamily.MATRIX),
    "linear": OperatorSpec("linear", OperatorFamily.MATRIX),
}


def _get_stream(device):
    if isinstance(device, torch.device):
        device = device.type
    if isinstance(device, str) and ":" in device:
        device = device.split(":")[0]
    if device == "cpu":
        return 0

    mod_name, attr = INFINIOPS_STREAM_ACCESSORS.get(device, (None, None))
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


def _random_strided(shape, strides, *, dtype=None, device=None, nonzero=False):
    out = _empty_strided(shape, strides, dtype=dtype, device=device)
    flat = out.as_strided((out.untyped_storage().size() // out.element_size(),), (1,))
    if flat.is_floating_point():
        if nonzero:
            flat.uniform_(1.0, 10.0)
        else:
            flat.normal_()
    else:
        flat.random_(1 if nonzero else -9, 10)
    return out


def _clone_strided(inp):
    out = _empty_strided(inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device)
    flat_args = (out.untyped_storage().size() // out.element_size(),), (1,)
    out.as_strided(*flat_args).copy_(inp.as_strided(*flat_args))
    return out


def _clone(obj):
    if isinstance(obj, torch.Tensor):
        return _clone_strided(obj)
    if isinstance(obj, tuple):
        return tuple(_clone(arg) for arg in obj)
    if isinstance(obj, list):
        return [_clone(arg) for arg in obj]
    if isinstance(obj, dict):
        return {key: _clone(value) for key, value in obj.items()}
    return obj


def _synchronize(device):
    if device == "cpu":
        return
    mod = getattr(torch, device, None)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def _get_attributes(config: dict) -> dict:
    """Convert the attribute list into a name-to-value mapping."""
    return {
        attr[AttributeSpec.NAME]: attr[AttributeSpec.VALUE]
        for attr in config.get(OperatorConfig.ATTRIBUTES, [])
    }


def _load_device_plugin(device: str) -> None:
    """Load the PyTorch extension that registers a vendor device."""
    module_name = INFINIOPS_DEVICE_PLUGIN_MODULES.get(device)
    if module_name is None:
        return

    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"PyTorch device plugin '{module_name}' is required for device "
            f"'{device}'"
        ) from exc


def _pick_slot(op_name: str, device: str) -> int:
    """Return the first implementation slot registered by InfiniOps."""
    op_pascal = "".join(part.capitalize() for part in op_name.split("_"))
    op_cls = getattr(infini.ops, op_pascal, None)
    if op_cls is None:
        raise ValueError(f"InfiniOps operator class not found: {op_pascal}")
    if not hasattr(op_cls, "active_implementation_indices"):
        raise RuntimeError(
            f"InfiniOps operator '{op_name}' cannot report implementations"
        )

    indices = op_cls.active_implementation_indices(device)
    if not indices:
        raise RuntimeError(
            f"InfiniOps operator '{op_name}' has no active implementation "
            f"for device '{device}'"
        )
    return indices[0]


def _tensor_specs(config: dict, field_name: str) -> List[Dict[str, Any]]:
    specs = config.get(field_name, [])
    if not isinstance(specs, list):
        raise ValueError(f"{field_name} must be a list")
    return specs


def _require_inputs(config: dict, count: int) -> List[Dict[str, Any]]:
    inputs = _tensor_specs(config, OperatorConfig.INPUTS)
    if len(inputs) < count:
        raise ValueError(f"operator requires at least {count} input tensor(s)")
    return inputs


def _output_shape(config: dict, fallback) -> Any:
    outputs = _tensor_specs(config, OperatorConfig.OUTPUTS)
    if outputs:
        return outputs[0][TensorSpec.SHAPE]
    return fallback


def _build_binary_case(
    spec: OperatorSpec, torch_device: str, torch_dtype, config: dict
) -> BenchmarkCase:
    inputs = _require_inputs(config, 2)
    shape = inputs[0][TensorSpec.SHAPE]
    out_shape = _output_shape(config, shape)
    a = _random_strided(shape, None, dtype=torch_dtype, device=torch_device)
    b = _random_strided(
        inputs[1][TensorSpec.SHAPE],
        None,
        dtype=torch_dtype,
        device=torch_device,
        nonzero=spec.name == "div",
    )
    out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)
    stream = _get_stream(torch_device)
    slot = _pick_slot(spec.name, torch_device)
    infini_op = getattr(infini.ops, spec.name)
    torch_op = getattr(torch, spec.torch_name)

    def operation(a, b, out):
        infini_op(
            a,
            b,
            *spec.scalar_args,
            out,
            stream=stream,
            implementation_index=slot,
        )
        return out

    def reference(a, b, out):
        if spec.name == "div" and not a.is_floating_point():
            torch_op(a, b, rounding_mode="trunc", out=out)
        else:
            torch_op(a, b, out=out)
        return out

    return BenchmarkCase(operation, reference, (a, b, out))


def _build_cast_case(
    spec: OperatorSpec, torch_device: str, torch_dtype, config: dict
) -> BenchmarkCase:
    inputs = _require_inputs(config, 1)
    outputs = _tensor_specs(config, OperatorConfig.OUTPUTS)
    shape = inputs[0][TensorSpec.SHAPE]
    out_dtype_name = (
        outputs[0].get(TensorSpec.DTYPE, "float32") if outputs else "float32"
    )
    if out_dtype_name not in _DTYPE_MAP:
        raise ValueError(f"Unsupported output dtype: {out_dtype_name}")

    inp = _random_strided(shape, None, dtype=torch_dtype, device=torch_device)
    out = _empty_strided(
        shape,
        None,
        dtype=_DTYPE_MAP[out_dtype_name],
        device=torch_device,
    )
    stream = _get_stream(torch_device)
    slot = _pick_slot(spec.name, torch_device)

    def operation(inp, out):
        infini.ops.cast(inp, out, stream=stream, implementation_index=slot)
        return out

    def reference(inp, out):
        out.copy_(inp.to(out.dtype))
        return out

    return BenchmarkCase(operation, reference, (inp, out))


def _build_concat_case(
    spec: OperatorSpec, torch_device: str, torch_dtype, config: dict
) -> BenchmarkCase:
    inputs = _require_inputs(config, 1)
    dim = _get_attributes(config).get("dim", 0)
    tensors = tuple(
        _random_strided(
            input_spec[TensorSpec.SHAPE],
            None,
            dtype=torch_dtype,
            device=torch_device,
        )
        for input_spec in inputs
    )
    fallback_shape = list(tensors[0].shape)
    fallback_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    out = _empty_strided(
        _output_shape(config, fallback_shape),
        None,
        dtype=torch_dtype,
        device=torch_device,
    )
    stream = _get_stream(torch_device)
    slot = _pick_slot(spec.name, torch_device)

    def operation(*args):
        inps = list(args[:-1])
        output = args[-1]
        infini.ops.cat(
            inps,
            dim,
            output,
            stream=stream,
            implementation_index=slot,
        )
        return output

    def reference(*args):
        inps = list(args[:-1])
        output = args[-1]
        output.copy_(torch.cat(inps, dim=dim))
        return output

    return BenchmarkCase(operation, reference, (*tensors, out))


def _logical_matrix(tensor, transpose: bool):
    return tensor.transpose(-2, -1) if transpose else tensor


def _build_matrix_case(
    spec: OperatorSpec, torch_device: str, torch_dtype, config: dict
) -> BenchmarkCase:
    inputs = _require_inputs(config, 2)
    attrs = _get_attributes(config)
    trans_a = bool(attrs.get("trans_a", False))
    trans_b = bool(attrs.get("trans_b", False))
    a_shape = inputs[0][TensorSpec.SHAPE]
    b_shape = inputs[1][TensorSpec.SHAPE]
    fallback_shape = [*a_shape[:-1], b_shape[-1]]
    out_shape = _output_shape(config, fallback_shape)

    a = _random_strided(a_shape, None, dtype=torch_dtype, device=torch_device)
    b = _random_strided(b_shape, None, dtype=torch_dtype, device=torch_device)
    stream = _get_stream(torch_device)
    slot = _pick_slot(spec.name, torch_device)

    if spec.name == "gemm":
        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 0.0)
        out = _random_strided(out_shape, None, dtype=torch_dtype, device=torch_device)

        def operation(a, b, alpha, beta, trans_a, trans_b, out):
            infini.ops.gemm(
                a,
                b,
                alpha,
                beta,
                trans_a,
                trans_b,
                out,
                stream=stream,
                implementation_index=slot,
            )
            return out

        def reference(a, b, alpha, beta, trans_a, trans_b, out):
            if alpha == 0:
                out.mul_(beta)
                return out
            product = torch.matmul(
                _logical_matrix(a, trans_a).float(),
                _logical_matrix(b, trans_b).float(),
            )
            out.copy_((alpha * product + beta * out.float()).to(out.dtype))
            return out

        args = (a, b, alpha, beta, trans_a, trans_b, out)
        return BenchmarkCase(operation, reference, args)

    out = _empty_strided(out_shape, None, dtype=torch_dtype, device=torch_device)

    if spec.name == "linear":
        has_bias = bool(attrs.get("has_bias", len(inputs) >= 3))
        bias = None
        if has_bias:
            bias_shape = (
                inputs[2][TensorSpec.SHAPE] if len(inputs) >= 3 else (out_shape[-1],)
            )
            bias = _random_strided(
                bias_shape, None, dtype=torch_dtype, device=torch_device
            )

        def operation(a, b, bias, out):
            infini.ops.linear(
                a,
                b,
                bias,
                trans_a,
                trans_b,
                out,
                stream=stream,
                implementation_index=slot,
            )
            return out

        def reference(a, b, bias, out):
            result = torch.matmul(
                _logical_matrix(a, trans_a).float(),
                _logical_matrix(b, trans_b).float(),
            )
            if bias is not None:
                result = result + bias.float()
            out.copy_(result.to(out.dtype))
            return out

        return BenchmarkCase(operation, reference, (a, b, bias, out))

    if spec.name == "matmul":

        def operation(a, b, out):
            infini.ops.matmul(
                a,
                b,
                out,
                trans_a,
                trans_b,
                stream=stream,
                implementation_index=slot,
            )
            return out

        def reference(a, b, out):
            result = torch.matmul(
                _logical_matrix(a, trans_a).float(),
                _logical_matrix(b, trans_b).float(),
            )
            out.copy_(result.to(out.dtype))
            return out

        return BenchmarkCase(operation, reference, (a, b, out))

    def operation(a, b, out):
        infini.ops.mm(a, b, out, stream=stream, implementation_index=slot)
        return out

    def reference(a, b, out):
        out.copy_(torch.mm(a.float(), b.float()).to(out.dtype))
        return out

    return BenchmarkCase(operation, reference, (a, b, out))


CASE_BUILDERS: Dict[str, Callable[..., BenchmarkCase]] = {
    OperatorFamily.BINARY: _build_binary_case,
    OperatorFamily.CAST: _build_cast_case,
    OperatorFamily.CONCAT: _build_concat_case,
    OperatorFamily.MATRIX: _build_matrix_case,
}


class InfiniOpsAdapter(BaseAdapter):
    """Adapter for common InfiniOps operator performance tests."""

    def __init__(self):
        self._req_metrics_template = []

    def process(self, test_input: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        test_input = self._normalize_test_input(test_input)
        if not test_input:
            raise ValueError(f"Invalid test_input type: {type(test_input)}")

        testcase = test_input.get(InfiniBenchJson.TESTCASE, "unknown")
        logger.info(f"InfiniOpsAdapter: Processing {testcase}")
        config = test_input.get(InfiniBenchJson.CONFIG, {})
        self._req_metrics_template = test_input.get(InfiniBenchJson.METRICS, [])

        try:
            operator_name = str(config.get(OperatorConfig.OPERATOR, "")).lower()
            operator_spec = OPERATOR_SPECS.get(operator_name)
            if operator_spec is None:
                return self._create_error_response(
                    f"Unsupported operator: {operator_name or '<empty>'}",
                    test_input,
                )

            platform = str(config.get(OperatorConfig.DEVICE, "cpu")).lower()
            torch_device = INFINIOPS_PLATFORM_TO_TORCH_DEVICE.get(platform, platform)
            _load_device_plugin(torch_device)

            inputs = _tensor_specs(config, OperatorConfig.INPUTS)
            dtype_name = (
                str(inputs[0].get(TensorSpec.DTYPE, "float16")).lower()
                if inputs
                else "float16"
            )
            if dtype_name not in _DTYPE_MAP:
                raise ValueError(f"Unsupported dtype: {dtype_name}")

            warmup = config.get(
                OperatorConfig.WARMUP_ITERATIONS,
                DEFAULT_WARMUP_ITERATIONS,
            )
            measured = config.get(
                OperatorConfig.MEASURED_ITERATIONS,
                DEFAULT_MEASURED_ITERATIONS,
            )
            self._validate_iterations(warmup, measured)
            self._check_op_available(operator_name)

            benchmark_case = self._build_case(
                operator_spec,
                torch_device,
                _DTYPE_MAP[dtype_name],
                config,
            )
            tolerance = config.get(OperatorConfig.TOLERANCE, DEFAULT_TOLERANCE)
            avg_latency_s, accuracy_pass = self._run_benchmark(
                benchmark_case,
                torch_device,
                warmup,
                measured,
                tolerance,
            )
            return self._create_response(
                test_input,
                config,
                operator_name,
                avg_latency_s,
                accuracy_pass,
            )
        except Exception as exc:
            operator = config.get(OperatorConfig.OPERATOR, "unknown")
            device = config.get(OperatorConfig.DEVICE, "unknown")
            logger.error(
                f"InfiniOpsAdapter: Operator test failed for {testcase}\n"
                f"  Operator: {operator}\n  Device: {device}\n  Error: {exc}",
                exc_info=True,
            )
            raise

    @staticmethod
    def _validate_iterations(warmup: Any, measured: Any) -> None:
        if not isinstance(warmup, int) or warmup < 0:
            raise ValueError("warmup_iterations must be a non-negative integer")
        if not isinstance(measured, int) or measured <= 0:
            raise ValueError("measured_iterations must be a positive integer")

    @staticmethod
    def _check_op_available(operator_name: str) -> None:
        op_pascal = "".join(part.capitalize() for part in operator_name.split("_"))
        if getattr(infini.ops, op_pascal, None) is None:
            raise ValueError(f"InfiniOps operator class not found: {op_pascal}")

    @staticmethod
    def _build_case(
        spec: OperatorSpec, torch_device: str, torch_dtype, config: dict
    ) -> BenchmarkCase:
        builder = CASE_BUILDERS[spec.family]
        return builder(spec, torch_device, torch_dtype, config)

    def _run_benchmark(
        self,
        benchmark_case: BenchmarkCase,
        torch_device: str,
        warmup: int,
        measured: int,
        tolerance: dict,
    ) -> Tuple[float, bool]:
        cloned_args = _clone(benchmark_case.args)
        cloned_kwargs = _clone(benchmark_case.kwargs)
        output = benchmark_case.operation(*benchmark_case.args, **benchmark_case.kwargs)
        expected = benchmark_case.reference(*cloned_args, **cloned_kwargs)
        accuracy_pass = self._outputs_match(output, expected, tolerance)

        for _ in range(warmup):
            benchmark_case.operation(*benchmark_case.args, **benchmark_case.kwargs)
        _synchronize(torch_device)

        total_time = 0.0
        for _ in range(measured):
            _synchronize(torch_device)
            start = time.perf_counter()
            benchmark_case.operation(*benchmark_case.args, **benchmark_case.kwargs)
            _synchronize(torch_device)
            total_time += time.perf_counter() - start

        return total_time / measured, accuracy_pass

    @staticmethod
    def _outputs_match(output: Any, expected: Any, tolerance: dict) -> bool:
        atol = tolerance.get("atol", DEFAULT_TOLERANCE["atol"])
        rtol = tolerance.get("rtol", DEFAULT_TOLERANCE["rtol"])

        if isinstance(output, tuple):
            if not isinstance(expected, tuple) or len(output) != len(expected):
                return False
            return all(
                InfiniOpsAdapter._tensor_matches(actual, reference, rtol, atol)
                for actual, reference in zip(output, expected)
            )
        if isinstance(output, torch.Tensor):
            return InfiniOpsAdapter._tensor_matches(output, expected, rtol, atol)
        return True

    @staticmethod
    def _tensor_matches(actual, expected, rtol: float, atol: float) -> bool:
        if actual.dtype.is_floating_point:
            return torch.allclose(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        return torch.equal(actual, expected)

    def _create_response(
        self,
        test_input: dict,
        config: dict,
        operator_name: str,
        avg_latency_s: float,
        accuracy_pass: bool,
    ) -> Dict[str, Any]:
        response = copy.deepcopy(test_input)
        response[InfiniBenchJson.TIME] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response[InfiniBenchJson.RESULT_CODE] = (
            ErrorCode.SUCCESS if accuracy_pass else ErrorCode.INTERNAL
        )
        if not accuracy_pass:
            response[
                InfiniBenchJson.ERROR_MSG
            ] = f"Accuracy check failed for operator '{operator_name}'"
        response[InfiniBenchJson.METRICS] = self._compute_metrics(
            config, avg_latency_s, accuracy_pass
        )
        return response

    def _compute_metrics(
        self, config: dict, avg_latency_s: float, accuracy_pass: bool
    ) -> List[Dict]:
        metrics = copy.deepcopy(self._req_metrics_template)
        inputs = config.get(OperatorConfig.INPUTS, [])
        outputs = config.get(OperatorConfig.OUTPUTS, [])
        operator = config.get(OperatorConfig.OPERATOR, "").lower()

        for metric in metrics:
            name = metric.get(MetricSpec.NAME, "")
            if name == OperatorMetric.LATENCY:
                self._set_scalar_metric(metric, round(avg_latency_s * 1000, 6), "ms")
            elif name == OperatorMetric.ACCURACY:
                metric.update(
                    {
                        MetricSpec.VALUE: "PASS" if accuracy_pass else "FAIL",
                        MetricSpec.UNIT: "",
                    }
                )
            elif name == OperatorMetric.FLOPS:
                value = 0.0
                if avg_latency_s > 0:
                    flops = FLOPSCalculator.get_flops(operator, inputs, outputs)
                    if flops > 0:
                        value = (flops / avg_latency_s) / 1e12
                        if value >= 0.0001:
                            value = round(value, 4)
                self._set_scalar_metric(metric, value, "TFLOPS")
            elif name == OperatorMetric.BANDWIDTH:
                value = 0.0
                if avg_latency_s > 0:
                    bandwidth = calculate_bandwidth(inputs, outputs)
                    total_bytes = bandwidth[BandwidthField.TOTAL_BYTES]
                    if total_bytes > 0:
                        value = (total_bytes / avg_latency_s) / 1e9
                        if value >= 0.0001:
                            value = round(value, 4)
                self._set_scalar_metric(metric, value, "GB/s")

        return metrics

    @staticmethod
    def _set_scalar_metric(metric: dict, value: Any, unit: str) -> None:
        metric.update(
            {
                MetricSpec.VALUE: value,
                MetricSpec.TYPE: MetricType.SCALAR,
                MetricSpec.RAW_DATA_URL: "",
                MetricSpec.UNIT: unit,
            }
        )

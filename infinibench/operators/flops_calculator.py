#!/usr/bin/env python3
"""
FLOPS and Bandwidth Calculator for InfiniCore Operators

Note: All values are theoretical estimates based on mathematical formulas.
Actual performance may vary due to hardware optimizations and implementation details.
"""

from typing import Dict, List, Optional, Callable

from infinibench.common.constants import DTYPE_BYTES_MAP


class FLOPSCalculator:
    """Calculate FLOPS for various operators using registry pattern"""

    # Registry for operator-specific FLOPS calculators
    _flops_registry: Dict[str, Callable[[List[Dict], List[Dict]], float]] = {}

    @classmethod
    def register(cls, operator_names: List[str]):
        """
        Decorator to register a FLOPS calculator for one or more operators.

        Usage:
            @FLOPSCalculator.register(["matmul", "matmul_tt"])
            def _matmul_flops(inputs, outputs):
                ...
        """

        def decorator(func: Callable[[List[Dict], List[Dict]], float]):
            for name in operator_names:
                cls._flops_registry[name.lower()] = func
            return func

        return decorator

    # FLOPS multipliers for different operators
    ACTIVATION_MULTIPLIERS = {
        "relu": 1.0,  # max(0, x)
        "sigmoid": 2.0,  # 1/(1+e^(-x))
        "tanh": 2.0,  # (e^x-e^(-x))/(e^x+e^(-x))
        "gelu": 3.0,  # x*Φ(x)
        "silu": 3.0,  # x*sigmoid(x)
        "swish": 3.0,  # same as silu
    }

    NORM_MULTIPLIERS = {
        "layernorm": 6.0,  # mean, var, normalize, scale, shift
        "rmsnorm": 4.0,  # square mean, sqrt, normalize, scale
        "batchnorm": 4.0,  # mean, var, normalize, scale/shift
    }

    @staticmethod
    def get_flops(
        operator: str,
        inputs: List[Dict],
        outputs: List[Dict],
        kwargs: Optional[Dict] = None,
    ) -> float:
        """
        Get FLOPS for a given operator.

        Args:
            operator: Operator name (e.g., 'matmul', 'conv2d', 'add')
            inputs: List of input tensor specs with 'shape' and 'dtype'
            outputs: List of output tensor specs
            kwargs: Operator-specific kwargs

        Returns:
            FLOPS count as float
        """
        op = operator.lower()
        size = FLOPSCalculator._get_tensor_size(outputs[0] if outputs else inputs[0])

        # Check registry first
        if op in FLOPSCalculator._flops_registry:
            return FLOPSCalculator._flops_registry[op](inputs, outputs)

        # Activation functions
        if op in FLOPSCalculator.ACTIVATION_MULTIPLIERS:
            return size * FLOPSCalculator.ACTIVATION_MULTIPLIERS[op]

        # Normalization functions
        if op in FLOPSCalculator.NORM_MULTIPLIERS:
            return size * FLOPSCalculator.NORM_MULTIPLIERS[op]

        # Special operators
        special_ops = {
            "softmax": 5.0,  # exp, sum, div
            "swiglu": 4.0,  # SiLU (3) + mul (1)
            "causalsoftmax": 2.5,  # masked softmax (~half elements)
        }
        if op in special_ops:
            return size * special_ops[op]

        if op == "embedding":
            return 0.0  # memory lookup only

        # Default: element-wise operation
        return float(size)

    @staticmethod
    def _get_tensor_size(tensor: Dict) -> int:
        """Get total number of elements in tensor"""
        shape = tensor.get("shape", [])
        size = 1
        for dim in shape:
            size *= dim
        return size


# Register matrix operations
@FLOPSCalculator.register(["matmul", "bmm", "batchmm"])
def _matmul_flops(inputs: List[Dict], outputs: List[Dict]) -> float:
    """Matrix Multiplication: C = A @ B (FLOPS = 2 * M * N * K)"""
    if len(inputs) < 2:
        return 0.0

    a_shape = inputs[0].get("shape", [])
    b_shape = inputs[1].get("shape", [])

    if len(a_shape) == 2 and len(b_shape) == 2:
        m, k = a_shape
        k2, n = b_shape
        return 2.0 * m * n * k
    elif len(a_shape) >= 2 and len(b_shape) >= 2:
        m, k = a_shape[-2], a_shape[-1]
        k2, n = b_shape[-2], b_shape[-1]
        batch_size = 1
        for dim in a_shape[:-2]:
            batch_size *= dim
        return 2.0 * batch_size * m * n * k

    return 0.0


@FLOPSCalculator.register(["addmm", "linear"])
def _addmm_flops(inputs: List[Dict], outputs: List[Dict]) -> float:
    """AddMM: C = beta * bias + alpha * (input @ weight)"""
    if len(inputs) < 3:
        return 0.0

    input_shape = inputs[1].get("shape", [])
    weight_shape = inputs[2].get("shape", [])

    if len(input_shape) >= 2 and len(weight_shape) >= 2:
        m, k = input_shape[-2], input_shape[-1]
        k2, n = weight_shape[-2], weight_shape[-1]

        matmul_flops = 2.0 * m * n * k
        output_size = m * n
        for dim in input_shape[:-2]:
            output_size *= dim

        return matmul_flops + output_size

    return 0.0


@FLOPSCalculator.register(["conv2d", "conv2d_backward"])
def _conv2d_flops(inputs: List[Dict], outputs: List[Dict]) -> float:
    """
    2D Convolution (FLOPS = 2 * N * C_out * H_out * W_out * K_h * K_w * C_in / groups)

    Note:
    - Uses output shape to correctly account for stride and padding effects.
    """
    if len(inputs) < 2 or not outputs:
        return 0.0

    # Input: [N, C_in, H_in, W_in]
    input_shape = inputs[0].get("shape", [])
    # Weight: [C_out, C_in, K_h, K_w]
    weight_shape = inputs[1].get("shape", [])
    # Output: [N, C_out, H_out, W_out]
    output_shape = outputs[0].get("shape", [])

    if len(input_shape) != 4 or len(weight_shape) != 4 or len(output_shape) != 4:
        return 0.0

    # Get kernel size
    kh, kw = weight_shape[2], weight_shape[3]
    c_in = input_shape[1]  # or weight_shape[1]

    # TODO: Currently assumes groups=1 (standard convolution). For group convolution,
    # FLOPS should be divided by groups. Update to extract groups from config/kwargs
    # when InfiniCore supports it.
    groups = 1

    # Key fix: use output tensor's H and W (accounts for stride/padding)
    n, c_out, h_out, w_out = output_shape

    return 2.0 * n * c_out * h_out * w_out * kh * kw * c_in / groups


def calculate_bandwidth(
    inputs: List[Dict],
    outputs: List[Dict],
) -> Dict[str, int]:
    """
    Calculate memory bandwidth requirements (bytes read/written).

    Args:
        inputs: List of input tensor specs
        outputs: List of output tensor specs

    Returns:
        Dict with 'read_bytes', 'write_bytes', 'total_bytes'

    Note:
        These are theoretical memory transfer sizes.
        Actual bandwidth may vary due to caching, memory alignment, and hardware optimizations.
    """

    def get_tensor_bytes(tensor: Dict) -> int:
        dtype = tensor.get("dtype", "float32").lower()
        bytes_per_element = DTYPE_BYTES_MAP.get(dtype, 4)
        size = FLOPSCalculator._get_tensor_size(tensor)
        return size * bytes_per_element

    read_bytes = sum(get_tensor_bytes(inp) for inp in inputs)
    write_bytes = sum(get_tensor_bytes(out) for out in outputs)

    return {
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "total_bytes": read_bytes + write_bytes,
    }

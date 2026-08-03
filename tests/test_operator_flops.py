from infinibench.operators.flops_calculator import FLOPSCalculator


def test_mm_flops_uses_matrix_formula():
    inputs = [
        {"shape": [2, 3], "dtype": "float16"},
        {"shape": [3, 4], "dtype": "float16"},
    ]

    assert FLOPSCalculator.get_flops("mm", inputs, [{"shape": [2, 4]}]) == 48


def test_linear_flops_uses_input_weight_and_bias():
    inputs = [
        {"shape": [2, 3], "dtype": "float16"},
        {"shape": [3, 4], "dtype": "float16"},
        {"shape": [4], "dtype": "float16"},
    ]

    assert FLOPSCalculator.get_flops("linear", inputs, [{"shape": [2, 4]}]) == 56

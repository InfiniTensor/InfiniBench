import importlib
import sys
import types
from unittest.mock import Mock, call

import pytest


@pytest.fixture
def adapter_module(monkeypatch):
    fake_torch = types.ModuleType("torch")

    class FakeDevice:
        def __init__(self, device_type="cpu"):
            self.type = device_type

    class FakeTensor:
        pass

    fake_torch.device = FakeDevice
    fake_torch.Tensor = FakeTensor
    fake_torch.float32 = object()
    fake_torch.float16 = object()
    fake_torch.bfloat16 = object()
    fake_torch.int64 = object()
    fake_torch.int32 = object()
    fake_torch.int16 = object()
    fake_torch.int8 = object()

    fake_infini = types.ModuleType("infini")
    fake_infini.__path__ = []
    fake_ops = types.ModuleType("infini.ops")
    fake_infini.ops = fake_ops

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "infini", fake_infini)
    monkeypatch.setitem(sys.modules, "infini.ops", fake_ops)
    monkeypatch.delitem(
        sys.modules, "infinibench.operators.infiniops_adapter", raising=False
    )

    return importlib.import_module("infinibench.operators.infiniops_adapter")


def test_pick_slot_preserves_native_slot_zero(adapter_module):
    class Add:
        @staticmethod
        def active_implementation_indices(device):
            return [0, 8]

    adapter_module.infini.ops.Add = Add

    assert adapter_module._pick_slot("add", "mlu") == 0


def test_pick_slot_preserves_registered_aten_fallback(adapter_module):
    class Add:
        @staticmethod
        def active_implementation_indices(device):
            return [8]

    adapter_module.infini.ops.Add = Add

    assert adapter_module._pick_slot("add", "mlu") == 8


def test_pick_slot_rejects_unregistered_fallback(adapter_module):
    class Add:
        @staticmethod
        def active_implementation_indices(device):
            return []

    adapter_module.infini.ops.Add = Add

    with pytest.raises(RuntimeError, match="no active implementation.*mlu"):
        adapter_module._pick_slot("add", "mlu")


@pytest.mark.parametrize("dtype_name,is_floating", [("float16", True), ("int8", False)])
def test_random_strided_initializes_storage_by_dtype(
    adapter_module, monkeypatch, dtype_name, is_floating
):
    flat = Mock()
    flat.is_floating_point.return_value = is_floating
    tensor = Mock()
    tensor.untyped_storage.return_value.size.return_value = 4
    tensor.element_size.return_value = 1
    tensor.as_strided.return_value = flat
    empty_strided = Mock(return_value=tensor)
    monkeypatch.setattr(adapter_module, "_empty_strided", empty_strided)

    dtype = getattr(adapter_module.torch, dtype_name)
    result = adapter_module._random_strided([2, 2], None, dtype=dtype, device="cpu")

    assert result is tensor
    tensor.as_strided.assert_called_once_with((4,), (1,))
    if is_floating:
        flat.normal_.assert_called_once_with()
        flat.random_.assert_not_called()
    else:
        flat.random_.assert_called_once_with(-9, 10)
        flat.normal_.assert_not_called()


@pytest.mark.parametrize("dtype_name,is_floating", [("float16", True), ("int8", False)])
def test_random_strided_generates_nonzero_values(
    adapter_module, monkeypatch, dtype_name, is_floating
):
    flat = Mock()
    flat.is_floating_point.return_value = is_floating
    tensor = Mock()
    tensor.untyped_storage.return_value.size.return_value = 4
    tensor.element_size.return_value = 1
    tensor.as_strided.return_value = flat
    monkeypatch.setattr(adapter_module, "_empty_strided", Mock(return_value=tensor))

    adapter_module._random_strided(
        [2, 2],
        None,
        dtype=getattr(adapter_module.torch, dtype_name),
        device="cpu",
        nonzero=True,
    )

    if is_floating:
        flat.uniform_.assert_called_once_with(1.0, 10.0)
        flat.random_.assert_not_called()
    else:
        flat.random_.assert_called_once_with(1, 10)
        flat.uniform_.assert_not_called()
    flat.normal_.assert_not_called()


def test_integer_cast_case_builds_and_forwards_slot(adapter_module, monkeypatch):
    inp = object()
    out = object()
    random_strided = Mock(return_value=inp)
    empty_strided = Mock(return_value=out)
    pick_slot = Mock(return_value=4)
    cast = Mock()

    monkeypatch.setattr(adapter_module, "_random_strided", random_strided)
    monkeypatch.setattr(adapter_module, "_empty_strided", empty_strided)
    monkeypatch.setattr(adapter_module, "_get_stream", lambda device: 123)
    monkeypatch.setattr(adapter_module, "_pick_slot", pick_slot)
    monkeypatch.setattr(adapter_module.infini.ops, "cast", cast, raising=False)

    config = {
        adapter_module.OperatorConfig.INPUTS: [
            {
                adapter_module.TensorSpec.SHAPE: [2, 2],
                adapter_module.TensorSpec.DTYPE: "int8",
            }
        ],
        adapter_module.OperatorConfig.OUTPUTS: [
            {
                adapter_module.TensorSpec.SHAPE: [2, 2],
                adapter_module.TensorSpec.DTYPE: "float32",
            }
        ],
    }
    case = adapter_module._build_cast_case(
        adapter_module.OPERATOR_SPECS["cast"],
        "cuda",
        adapter_module.torch.int8,
        config,
    )

    random_strided.assert_called_once_with(
        [2, 2], None, dtype=adapter_module.torch.int8, device="cuda"
    )
    empty_strided.assert_called_once_with(
        [2, 2], None, dtype=adapter_module.torch.float32, device="cuda"
    )
    pick_slot.assert_called_once_with("cast", "cuda")

    assert case.operation(*case.args) is out
    cast.assert_called_once_with(inp, out, stream=123, implementation_index=4)


def test_concat_case_selects_and_forwards_slot(adapter_module, monkeypatch):
    first = types.SimpleNamespace(shape=(2, 3))
    second = types.SimpleNamespace(shape=(2, 4))
    out = object()
    random_strided = Mock(side_effect=(first, second))
    pick_slot = Mock(return_value=7)
    cat = Mock()

    monkeypatch.setattr(adapter_module, "_random_strided", random_strided)
    monkeypatch.setattr(adapter_module, "_empty_strided", lambda *args, **kwargs: out)
    monkeypatch.setattr(adapter_module, "_get_stream", lambda device: 456)
    monkeypatch.setattr(adapter_module, "_pick_slot", pick_slot)
    monkeypatch.setattr(adapter_module.infini.ops, "cat", cat, raising=False)

    config = {
        adapter_module.OperatorConfig.INPUTS: [
            {adapter_module.TensorSpec.SHAPE: [2, 3]},
            {adapter_module.TensorSpec.SHAPE: [2, 4]},
        ],
        adapter_module.OperatorConfig.OUTPUTS: [
            {adapter_module.TensorSpec.SHAPE: [2, 7]}
        ],
        adapter_module.OperatorConfig.ATTRIBUTES: [
            {
                adapter_module.AttributeSpec.NAME: "dim",
                adapter_module.AttributeSpec.VALUE: 1,
            }
        ],
    }
    case = adapter_module._build_concat_case(
        adapter_module.OPERATOR_SPECS["cat"],
        "mlu",
        adapter_module.torch.float16,
        config,
    )

    pick_slot.assert_called_once_with("cat", "mlu")
    assert case.operation(*case.args) is out
    cat.assert_called_once_with(
        [first, second],
        1,
        out,
        stream=456,
        implementation_index=7,
    )


def test_integer_div_uses_nonzero_divisor_and_truncating_reference(
    adapter_module, monkeypatch
):
    first = Mock()
    first.is_floating_point.return_value = False
    second = Mock()
    out = object()
    random_strided = Mock(side_effect=(first, second))
    torch_div = Mock()

    monkeypatch.setattr(adapter_module, "_random_strided", random_strided)
    monkeypatch.setattr(adapter_module, "_empty_strided", Mock(return_value=out))
    monkeypatch.setattr(adapter_module, "_get_stream", lambda device: 123)
    monkeypatch.setattr(adapter_module, "_pick_slot", Mock(return_value=4))
    monkeypatch.setattr(adapter_module.torch, "div", torch_div, raising=False)
    monkeypatch.setattr(adapter_module.infini.ops, "div", Mock(), raising=False)

    config = {
        adapter_module.OperatorConfig.INPUTS: [
            {adapter_module.TensorSpec.SHAPE: [2, 2]},
            {adapter_module.TensorSpec.SHAPE: [2, 2]},
        ],
        adapter_module.OperatorConfig.OUTPUTS: [
            {adapter_module.TensorSpec.SHAPE: [2, 2]}
        ],
    }
    case = adapter_module._build_binary_case(
        adapter_module.OPERATOR_SPECS["div"],
        "cuda",
        adapter_module.torch.int8,
        config,
    )

    assert random_strided.call_args_list == [
        call([2, 2], None, dtype=adapter_module.torch.int8, device="cuda"),
        call(
            [2, 2],
            None,
            dtype=adapter_module.torch.int8,
            device="cuda",
            nonzero=True,
        ),
    ]
    assert case.reference(*case.args) is out
    torch_div.assert_called_once_with(first, second, rounding_mode="trunc", out=out)


def test_only_common_operators_are_registered(adapter_module):
    assert set(adapter_module.OPERATOR_SPECS) == {
        "add",
        "sub",
        "mul",
        "div",
        "cast",
        "cat",
        "gemm",
        "matmul",
        "mm",
        "linear",
    }


def test_operator_specs_use_four_builder_families(adapter_module):
    assert {spec.family for spec in adapter_module.OPERATOR_SPECS.values()} == {
        adapter_module.OperatorFamily.BINARY,
        adapter_module.OperatorFamily.CAST,
        adapter_module.OperatorFamily.CONCAT,
        adapter_module.OperatorFamily.MATRIX,
    }
    assert set(adapter_module.CASE_BUILDERS) == {
        adapter_module.OperatorFamily.BINARY,
        adapter_module.OperatorFamily.CAST,
        adapter_module.OperatorFamily.CONCAT,
        adapter_module.OperatorFamily.MATRIX,
    }


def test_deferred_model_operators_are_not_registered(adapter_module):
    deferred = {
        "rms_norm",
        "causal_softmax",
        "swiglu",
        "flash_attention",
        "rotary_embedding",
        "add_rms_norm",
        "reshape_and_cache",
    }

    assert deferred.isdisjoint(adapter_module.OPERATOR_SPECS)


def test_deferred_operator_returns_structured_error(adapter_module):
    result = adapter_module.InfiniOpsAdapter().process(
        {
            "run_id": "deferred-op",
            "testcase": "operator.InfiniOps.FlashAttention",
            "config": {"operator": "flash_attention", "device": "cambricon"},
            "metrics": [],
        }
    )

    assert result["result_code"] != 0
    assert result["error_msg"] == "Unsupported operator: flash_attention"


def test_runtime_mappings_come_from_common_constants(adapter_module):
    assert adapter_module.INFINIOPS_PLATFORM_TO_TORCH_DEVICE["cambricon"] == "mlu"
    assert adapter_module.INFINIOPS_DEVICE_PLUGIN_MODULES["mlu"] == "torch_mlu"
    assert adapter_module.INFINIOPS_STREAM_ACCESSORS["mlu"] == (
        "mlu",
        "mlu_stream",
    )


def test_missing_device_plugin_has_actionable_error(adapter_module, monkeypatch):
    def missing_module(name):
        raise ImportError(name)

    monkeypatch.setattr(adapter_module.importlib, "import_module", missing_module)

    with pytest.raises(RuntimeError, match="torch_mlu.*mlu"):
        adapter_module._load_device_plugin("mlu")


def test_accuracy_failure_sets_nonzero_result_code(adapter_module, monkeypatch):
    adapter = adapter_module.InfiniOpsAdapter()
    monkeypatch.setitem(
        adapter_module.OPERATOR_SPECS,
        "test_op",
        adapter_module.OperatorSpec("test_op", adapter_module.OperatorFamily.BINARY),
    )
    monkeypatch.setattr(adapter, "_check_op_available", lambda *args: None)
    monkeypatch.setattr(
        adapter,
        "_build_case",
        lambda *args: adapter_module.BenchmarkCase(lambda: None, lambda: None, ()),
    )
    monkeypatch.setattr(adapter, "_run_benchmark", lambda *args: (0.001, False))

    result = adapter.process(
        {
            "run_id": "accuracy-failure",
            "testcase": "operator.infiniops.TestOp",
            "config": {
                "operator": "test_op",
                "device": "cpu",
                "inputs": [{"shape": [2, 2], "dtype": "float16"}],
                "outputs": [{"shape": [2, 2], "dtype": "float16"}],
                "warmup_iterations": 0,
                "measured_iterations": 1,
            },
            "metrics": [{"name": "operator.tensor_accuracy"}],
        }
    )

    assert result["result_code"] != 0
    assert result["error_msg"] == "Accuracy check failed for operator 'test_op'"
    assert result["metrics"][0]["value"] == "FAIL"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup_iterations", -1, "non-negative integer"),
        ("measured_iterations", 0, "positive integer"),
    ],
)
def test_iteration_counts_are_validated(
    adapter_module, monkeypatch, field, value, message
):
    adapter = adapter_module.InfiniOpsAdapter()
    monkeypatch.setattr(adapter_module, "_load_device_plugin", lambda device: None)
    config = {
        "operator": "add",
        "device": "cpu",
        "inputs": [{"shape": [1], "dtype": "float16"}],
        "outputs": [{"shape": [1], "dtype": "float16"}],
        "warmup_iterations": 0,
        "measured_iterations": 1,
    }
    config[field] = value

    with pytest.raises(ValueError, match=message):
        adapter.process(
            {
                "testcase": "operator.infiniops.Add",
                "config": config,
                "metrics": [],
            }
        )

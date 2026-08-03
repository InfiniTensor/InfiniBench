import importlib
import sys
import types

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
        sys.modules, "infinimetrics.operators.infiniops_adapter", raising=False
    )

    return importlib.import_module("infinimetrics.operators.infiniops_adapter")


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

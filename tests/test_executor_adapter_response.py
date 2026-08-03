import json
from pathlib import Path

from infinibench.adapter import BaseAdapter
from infinibench.common.constants import TestCategory as Category
from infinibench.dispatcher import Dispatcher, _ADAPTER_REGISTRY
from infinibench.executor import Executor


class ErrorResponseAdapter(BaseAdapter):
    def process(self, test_input):
        return {
            "run_id": test_input["run_id"],
            "testcase": test_input["testcase"],
            "result_code": 2,
            "error_msg": "accuracy failed",
            "config": test_input["config"],
            "metrics": [],
        }


def test_dispatcher_registers_infiniops_framework():
    assert Dispatcher()._parse_testcase("operator.InfiniOps.Add") == (
        "operator",
        "infiniops",
    )
    assert (Category.OPERATOR, "infiniops") in _ADAPTER_REGISTRY


def test_executor_propagates_adapter_error_response(tmp_path, monkeypatch):
    payload = {
        "run_id": "adapter-error",
        "testcase": "operator.InfiniOps.Add",
        "config": {"output_dir": str(tmp_path)},
        "metrics": [],
    }
    executor = Executor(payload, ErrorResponseAdapter())
    monkeypatch.setattr(executor, "_enrich_environment", lambda response: response)

    result = executor.execute()

    assert result.result_code == 2
    assert result.error_msg == "accuracy failed"
    saved = json.loads(Path(result.result_file).read_text(encoding="utf-8"))
    assert saved["result_code"] == 2

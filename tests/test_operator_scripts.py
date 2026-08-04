import json

import pytest

from scripts.aggregate_results import aggregate, discover_result_files
from scripts.generate_operator_inputs import (
    build_test_config,
    enumerate_test_cases,
    get_input_shapes,
    parse_args,
)


def test_generated_infiniops_config_matches_adapter_contract():
    config = build_test_config(
        operator="sub",
        device="cambricon",
        inputs=[
            {"name": "a", "shape": [4, 8], "dtype": "float16"},
            {"name": "b", "shape": [4, 8], "dtype": "float16"},
        ],
        output_shape=[4, 8],
        dtype="float16",
    )

    assert config["testcase"] == "operator.InfiniOps.Sub"
    assert config["config"]["device"] == "cambricon"
    assert config["config"]["tolerance"] == {"atol": 0.01, "rtol": 0.01}


def test_case_enumeration_is_deterministic():
    cases = enumerate_test_cases(["mm", "add"], ["small"], ["float16"])

    assert [case.index for case in cases] == list(range(len(cases)))
    assert [(case.operator, case.shape_label) for case in cases[:2]] == [
        ("mm", "64x64"),
        ("mm", "128x128"),
    ]


def test_linear_generator_uses_matrix_compatible_weight_shape():
    assert get_input_shapes("linear", (4, 8)) == [[4, 8], [8, 8], [8]]


@pytest.mark.parametrize("args", [["--measured", "0"], ["--warmup", "-1"]])
def test_cli_rejects_invalid_iteration_counts(args):
    with pytest.raises(SystemExit):
        parse_args(args)


def test_aggregate_falls_back_to_config_for_nonstandard_run_id(tmp_path):
    result_path = tmp_path / "custom_results.json"
    result_path.write_text(
        json.dumps(
            {
                "run_id": "custom-run",
                "testcase": "operator.InfiniOps.Add",
                "result_code": 0,
                "duration": 1.25,
                "config": {
                    "operator": "add",
                    "inputs": [{"dtype": "float16"}],
                },
                "metrics": [{"name": "operator.latency", "value": "unavailable"}],
            }
        ),
        encoding="utf-8",
    )

    summary = aggregate(
        discover_result_files(tmp_path),
        filter_operator="add",
        filter_dtype="float16",
    )

    assert summary["total"] == 1
    assert summary["by_operator"]["add"]["total"] == 1
    assert "latency_ms" not in summary["by_operator"]["add"]

from pathlib import Path

import pytest

from infinimetrics.dispatcher import Dispatcher
from infinimetrics.hardware import hardware_adapter
from infinimetrics.hardware.hardware_adapter import HardwareTestAdapter


CUDA_OUTPUT = """
Direction: Host to Device
Size (MB)    Time (ms)Bandwidth (GB/s)    CV (%)
-------------------------------------------------------
64.00        1.000       12.50             1.0

===================================================
STREAM Benchmark Suite
STREAM_Copy      100.00
STREAM_Scale      90.00
STREAM_Add        80.00
STREAM_Triad      70.00

L1 Cache Bandwidth Sweep Test
Eff. bw
-------------------------------------------------------
4 kB 1.0ms 0.1% 200.0GB/s
L2 Cache Bandwidth Sweep Test
Eff. bw
-------------------------------------------------------
256 kB 64 kB 1.0ms 0.1% 300.0GB/s
"""


def test_constructor_preserves_cuda_perf_path_and_default_layout(tmp_path):
    custom_binary = tmp_path / "custom_cuda_perf_suite"
    adapter = HardwareTestAdapter(str(custom_binary), output_dir=str(tmp_path))

    assert adapter.cuda_perf_path == str(custom_binary)
    assert adapter.output_dir == tmp_path
    assert adapter.build_dir.name == "cuda-memory-benchmark"
    assert adapter.build_script == adapter.build_dir / "build.sh"


def test_build_command_preserves_cuda_cli_shape(tmp_path):
    binary = tmp_path / "cuda_perf_suite"
    adapter = HardwareTestAdapter(str(binary), output_dir=str(tmp_path))

    command = adapter._build_command(
        {
            "test_type": "Comprehensive",
            "device_id": 2,
            "iterations": 7,
            "array_size": 1024,
        }
    )

    assert command == [
        str(binary),
        "--all",
        "--device",
        "2",
        "--iterations",
        "7",
        "--array-size",
        "1024",
    ]


def test_parse_stream_preserves_metric_names(tmp_path):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_stream_benchmark(CUDA_OUTPUT)

    assert metrics == [
        {
            "name": "hardware.stream_copy",
            "value": 100.0,
            "type": "scalar",
            "unit": "GB/s",
        },
        {
            "name": "hardware.stream_scale",
            "value": 90.0,
            "type": "scalar",
            "unit": "GB/s",
        },
        {
            "name": "hardware.stream_add",
            "value": 80.0,
            "type": "scalar",
            "unit": "GB/s",
        },
        {
            "name": "hardware.stream_triad",
            "value": 70.0,
            "type": "scalar",
            "unit": "GB/s",
        },
    ]


def test_parse_comprehensive_preserves_cuda_metric_names(tmp_path):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_output(CUDA_OUTPUT, "Comprehensive", "run-1")
    names = [metric["name"] for metric in metrics]

    assert names == [
        "hardware.mem_sweep_h2d",
        "hardware.stream_copy",
        "hardware.stream_scale",
        "hardware.stream_add",
        "hardware.stream_triad",
        "hardware.gpu_cache_l1",
        "hardware.gpu_cache_l2",
    ]
    assert Path(tmp_path, "mem_sweep_h2d_run-1_").parent == tmp_path
    assert any(tmp_path.glob("mem_sweep_h2d_run-1_*.csv"))
    assert any(tmp_path.glob("cache_l1_bandwidth_run-1_*.csv"))
    assert any(tmp_path.glob("cache_l2_bandwidth_run-1_*.csv"))


def test_parse_unknown_test_type_returns_no_metrics(tmp_path):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    assert adapter._parse_output(CUDA_OUTPUT, "Unknown", "run-1") == []


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda", "cuda"),
        ("nvidia", "cuda"),
        ("metax", "metax"),
        ("iluvatar", "corex"),
        ("hygon", "hygon"),
        ("moore", "moore"),
        ("musa", "moore"),
        ("cambricon", "cambricon"),
        ("mlu", "cambricon"),
        ("ascend", "ascend"),
        ("npu", "ascend"),
        ("legacy-unknown-device", "cuda"),
    ],
)
def test_explicit_device_aliases_preserve_unknown_cuda_fallback(
    tmp_path, device, expected
):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    assert adapter._get_device_type({"device": device}) == expected


def test_testcase_framework_precedes_runtime_detection(tmp_path, monkeypatch):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))
    monkeypatch.setattr(hardware_adapter, "detect_platform", lambda: "moore")

    assert (
        adapter._get_device_type({"_testcase": "hardware.iluvatar.Stream"})
        == "corex"
    )
    assert adapter._get_device_type({}) == "moore"


def test_native_platform_paths_do_not_change_cuda_path(tmp_path):
    cuda_binary = tmp_path / "cuda_perf_suite"
    adapter = HardwareTestAdapter(str(cuda_binary), output_dir=str(tmp_path))

    assert adapter._get_binary_path("cuda") == str(cuda_binary)
    assert adapter._get_binary_path("metax") == str(cuda_binary)
    assert adapter._get_binary_path("cambricon").endswith(
        "cambricon-memory-benchmark/build/mlu_perf_suite"
    )
    assert adapter._get_binary_path("ascend").endswith(
        "ascend-memory-benchmark/build/npu_perf_suite"
    )


@pytest.mark.parametrize(
    "framework",
    [
        "cudaunified",
        "cuda",
        "metax",
        "corex",
        "iluvatar",
        "hygon",
        "moore",
        "cambricon",
        "ascend",
    ],
)
def test_dispatcher_registers_every_hardware_framework(framework):
    adapter = Dispatcher()._create_adapter("hardware", framework)

    assert isinstance(adapter, HardwareTestAdapter)


def test_cambricon_cache_maps_to_existing_metric_names(tmp_path):
    output = """
NRAM Bandwidth Test (BANG Kernel)
NRAM chunk/core       Time (ms)  Eff. BW (GB/s)      TFLOPS    Spread
---------------------------------------------------------------------
120 kB                1.0        200.0                1.2       0.5%

===================================================
L2 Cache Bandwidth Sweep Test (BANG Kernel)
data set     exec data      exec time     spread       Eff. bw
---------------------------------------------------------------
256 kB       2560 kB        1ms            0.5%          300 GB/s
"""
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_output(output, "Cache", "cam-run", "cambricon")

    assert [metric["name"] for metric in metrics] == [
        "hardware.gpu_cache_l1",
        "hardware.gpu_cache_l2",
    ]


def test_ascend_hierarchy_is_not_relabelled_as_cuda_cache(tmp_path):
    output = """
Memory Hierarchy Sweep Test (D2D Bandwidth)
data set     exec data      exec time     spread       Eff. bw
---------------------------------------------------------------
256 kB       2560 kB        1ms            0.5%          300 GB/s
"""
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_output(output, "Cache", "ascend-run", "ascend")

    assert [metric["name"] for metric in metrics] == [
        "hardware.memory_hierarchy_d2d"
    ]


def test_native_benchmarks_use_selected_device_id():
    hardware_dir = Path(hardware_adapter.__file__).parent
    native_files = [
        *hardware_dir.glob("cambricon-memory-benchmark/include/*.h"),
        *hardware_dir.glob("ascend-memory-benchmark/include/*.h"),
    ]

    assert native_files
    for path in native_files:
        source = path.read_text(encoding="utf-8")
        assert "SetDevice(0)" not in source, path

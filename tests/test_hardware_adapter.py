from pathlib import Path

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

from pathlib import Path

import pytest

from infinibench.dispatcher import Dispatcher
from infinibench.hardware import hardware_adapter
from infinibench.hardware.hardware_adapter import HardwareTestAdapter

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


@pytest.mark.parametrize("header_spacing", ["", "  "])
def test_parse_memory_bandwidth_accepts_platform_header_spacing(
    tmp_path, header_spacing
):
    output = f"""
Direction: Host to Device
Size (MB)    Time (ms){header_spacing}Bandwidth (GB/s)    CV (%)
-------------------------------------------------------
64.00        1.000       12.50             1.0
"""
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_output(output, "MemSweep", "run-spacing")

    assert [metric["name"] for metric in metrics] == ["hardware.mem_sweep_h2d"]


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


def test_runtime_detection_ignores_testcase_framework(tmp_path, monkeypatch):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))
    monkeypatch.setattr(hardware_adapter, "detect_platform", lambda: "moore")

    assert (
        adapter._get_device_type({"_testcase": "hardware.cudaUnified.Stream"})
        == "moore"
    )
    assert adapter._get_device_type({}) == "moore"


@pytest.mark.parametrize("device", ["cuda", "metax", "corex", "hygon", "moore"])
def test_cuda_compatible_platforms_share_binary(tmp_path, device):
    cuda_binary = tmp_path / "cuda_perf_suite"
    adapter = HardwareTestAdapter(str(cuda_binary), output_dir=str(tmp_path))

    assert adapter._get_binary_path(device) == str(cuda_binary)


def test_cambricon_uses_native_binary_path(tmp_path):
    cuda_binary = tmp_path / "cuda_perf_suite"
    adapter = HardwareTestAdapter(str(cuda_binary), output_dir=str(tmp_path))

    assert Path(adapter._get_binary_path("cambricon")).parts[-3:] == (
        "cambricon-memory-benchmark",
        "build",
        "mlu_perf_suite",
    )
    assert adapter._get_binary_path("cuda") == str(cuda_binary)


def test_ascend_uses_native_binary_path(tmp_path):
    cuda_binary = tmp_path / "cuda_perf_suite"
    adapter = HardwareTestAdapter(str(cuda_binary), output_dir=str(tmp_path))

    assert Path(adapter._get_binary_path("ascend")).parts[-3:] == (
        "ascend-memory-benchmark",
        "build",
        "npu_perf_suite",
    )
    assert adapter._get_binary_path("cuda") == str(cuda_binary)


def test_build_cuda_project_preserves_runtime_platform_detection(tmp_path, monkeypatch):
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))
    built_platforms = []
    monkeypatch.setattr(hardware_adapter, "detect_platform", lambda: "metax")
    monkeypatch.setattr(adapter, "_build_project", built_platforms.append)

    adapter._build_cuda_project()

    assert built_platforms == ["metax"]


def test_dispatcher_registers_cudaunified_hardware_framework():
    adapter = Dispatcher()._create_adapter("hardware", "cudaunified")

    assert isinstance(adapter, HardwareTestAdapter)


@pytest.mark.parametrize(
    "device",
    [
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
def test_dispatcher_does_not_register_devices_as_frameworks(device):
    with pytest.raises(ValueError, match="Adapter not registered"):
        Dispatcher()._create_adapter("hardware", device)


def test_cambricon_nram_uses_platform_specific_metric_name(tmp_path):
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
        "hardware.nram_bandwidth",
        "hardware.gpu_cache_l2",
    ]


def test_ascend_d2d_size_sweep_has_copy_specific_metric_name(tmp_path):
    output = """
D2D Memcpy Size Sweep Test
data set     exec data      exec time     spread       Eff. bw
---------------------------------------------------------------
256 kB       2560 kB        1ms            0.5%          300 GB/s
"""
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_output(output, "Cache", "ascend-run", "ascend")

    assert [metric["name"] for metric in metrics] == ["hardware.d2d_memcpy_size_sweep"]


def test_ascend_benchmark_uses_selected_device_id():
    hardware_dir = Path(hardware_adapter.__file__).parent
    native_files = list(hardware_dir.glob("ascend-memory-benchmark/include/*.h"))

    assert native_files
    for path in native_files:
        source = path.read_text(encoding="utf-8")
        assert "SetDevice(0)" not in source, path


def test_cambricon_benchmark_uses_selected_device_id():
    hardware_dir = Path(hardware_adapter.__file__).parent
    native_files = list(hardware_dir.glob("cambricon-memory-benchmark/include/*.h"))

    assert native_files
    for path in native_files:
        source = path.read_text(encoding="utf-8")
        assert "SetDevice(0)" not in source, path


def test_cambricon_uses_current_cnrt_success_enum():
    source = (
        Path(hardware_adapter.__file__).parent
        / "cambricon-memory-benchmark"
        / "include"
        / "cnrt_utils.h"
    ).read_text(encoding="utf-8")

    assert "cnrtSuccess" in source
    assert "CNRT_RET_SUCCESS" not in source


def test_cambricon_nram_workload_and_read_volume_cover_every_core():
    source = (
        Path(hardware_adapter.__file__).parent
        / "cambricon-memory-benchmark"
        / "include"
        / "cache_benchmark.h"
    ).read_text(encoding="utf-8")

    assert "size_t total_elements = chunk * total_cores;" in source
    assert "(T*)dst, (const T*)src, total_elements" in source
    assert "double data_volume = 4.0 * chunk_bytes;" in source


def test_cambricon_bidirectional_copy_uses_distinct_host_buffers():
    source = (
        Path(hardware_adapter.__file__).parent
        / "cambricon-memory-benchmark"
        / "include"
        / "memory_bandwidth_test.h"
    ).read_text(encoding="utf-8")

    assert "cnrtMemcpyAsync(dev1, host_src, bytes, q1" in source
    assert "cnrtMemcpyAsync(host_dst, dev2, bytes, q2" in source


def test_cambricon_bandwidth_buffers_match_largest_sweep_case():
    source = (
        Path(hardware_adapter.__file__).parent
        / "cambricon-memory-benchmark"
        / "include"
        / "memory_bandwidth_test.h"
    ).read_text(encoding="utf-8")

    assert "const size_t max_bytes = sizes_kb.back() * 1024;" in source
    assert "2ULL * 1024 * 1024 * 1024" not in source


def test_ascend_memory_buffers_match_largest_sweep_case():
    source = (
        Path(hardware_adapter.__file__).parent
        / "ascend-memory-benchmark"
        / "include"
        / "memory_bandwidth_test.h"
    ).read_text(encoding="utf-8")

    assert "const size_t max_bytes = sizes_kb.back() * 1024;" in source
    assert "2ULL * 1024 * 1024 * 1024" not in source


def test_ascend_system_info_only_prints_selected_device():
    source = (
        Path(hardware_adapter.__file__).parent
        / "ascend-memory-benchmark"
        / "src"
        / "main.cc"
    ).read_text(encoding="utf-8")

    assert "NpuDeviceInfo::print(cfg.device_id);" in source
    assert "NpuDeviceInfo::print(i);" not in source


def test_ascend_stream_source_uses_device_arithmetic_operations():
    source = (
        Path(hardware_adapter.__file__).parent
        / "ascend-memory-benchmark"
        / "include"
        / "stream_benchmark.h"
    ).read_text(encoding="utf-8")

    assert '"STREAM_Copy"' in source
    assert '"STREAM_Scale"' in source
    assert '"STREAM_Add"' in source
    assert '"STREAM_Triad"' in source
    assert "aclnnMulsGetWorkspaceSize" in source
    assert "aclnnMuls" in source
    assert source.count("aclnnAddGetWorkspaceSize") == 2
    assert "AclScalarDescriptor add_alpha(1.0f)" in source
    assert "AclScalarDescriptor triad_alpha(3.0f)" in source
    assert "estimated" not in source.lower()


def test_ascend_stream_output_publishes_four_metrics(tmp_path):
    output = """
STREAM Benchmark Suite
Operation         Bandwidth (GB/s)     Time (ms)   CV (%)
----------------------------------------------------------
STREAM_Copy                 220.00          0.04      1.00
STREAM_Scale                210.00          0.04      1.00
STREAM_Add                  200.00          0.06      1.00
STREAM_Triad                190.00          0.06      1.00
"""
    adapter = HardwareTestAdapter(output_dir=str(tmp_path))

    metrics = adapter._parse_output(output, "Stream", "ascend-run", "ascend")

    assert [metric["name"] for metric in metrics] == [
        "hardware.stream_copy",
        "hardware.stream_scale",
        "hardware.stream_add",
        "hardware.stream_triad",
    ]


def test_ascend_bidirectional_copy_uses_distinct_host_buffers():
    source = (
        Path(hardware_adapter.__file__).parent
        / "ascend-memory-benchmark"
        / "include"
        / "memory_bandwidth_test.h"
    ).read_text(encoding="utf-8")

    assert "dev1.data(), max_bytes, host_src.data()" in source
    assert "host_dst.data(), max_bytes, dev2.data()" in source


def test_cambricon_cmake_uses_cncc_as_the_compiler():
    source = (
        Path(hardware_adapter.__file__).parent
        / "cambricon-memory-benchmark"
        / "CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert 'set(CMAKE_CXX_COMPILER "${CNCC}")' in source
    assert (
        "target_link_libraries(mlu_perf_suite ${CNRT_LIB} stdc++ m pthread)" in source
    )
    assert "CXX_COMPILER_LAUNCHER" not in source
    assert "RULE_LAUNCH_COMPILE" not in source


def test_cambricon_kernels_share_nram_layout_calculation():
    benchmark_dir = (
        Path(hardware_adapter.__file__).parent / "cambricon-memory-benchmark"
    )
    utility = (benchmark_dir / "include" / "nram_utils.h").read_text(encoding="utf-8")
    stream = (benchmark_dir / "include" / "stream_benchmark.h").read_text(
        encoding="utf-8"
    )
    cache = (benchmark_dir / "include" / "cache_benchmark.h").read_text(
        encoding="utf-8"
    )

    assert "prepare_nram_layout" in utility
    assert stream.count("prepare_nram_layout<T>") == 5
    assert cache.count("prepare_nram_layout<T>") == 2
    assert "#define NRAM_MAX" not in stream + cache


def test_cambricon_stream_measurement_uses_shared_control_flow():
    source = (
        Path(hardware_adapter.__file__).parent
        / "cambricon-memory-benchmark"
        / "include"
        / "stream_benchmark.h"
    ).read_text(encoding="utf-8")

    expected_cases = [
        'benchmark("STREAM_Copy", 2.0 * element_bytes',
        'benchmark("STREAM_Scale", 2.0 * element_bytes',
        'benchmark("STREAM_Add", 3.0 * element_bytes',
        'benchmark("STREAM_Triad", 3.0 * element_bytes',
    ]
    positions = [source.index(case) for case in expected_cases]

    assert positions == sorted(positions)
    assert source.count("cnrtNotifierCreate") == 2

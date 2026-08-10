import os
import subprocess
from pathlib import Path

import pytest

from infinibench.common.constants import CUDA_SAMPLE_CONFIGS
from infinibench.compatibility.compatibility_adapter import (
    _CUDA_SAMPLES_DIR,
    CompatibilityAdapter,
)
from infinibench.dispatcher import Dispatcher


def _sample(root: Path, category: str, name: str, manifest: str) -> Path:
    sample_dir = root / "Samples" / category / name
    sample_dir.mkdir(parents=True)
    content = f"project({name})\n" if manifest == "CMakeLists.txt" else "# test\n"
    (sample_dir / manifest).write_text(content, encoding="utf-8")
    return sample_dir


@pytest.fixture
def compiler_available(monkeypatch):
    monkeypatch.setattr(
        "infinibench.compatibility.compatibility_adapter.shutil.which",
        lambda candidate, **_: candidate,
    )


def test_default_cuda_samples_dir_uses_repository_submodule():
    repository_root = Path(__file__).resolve().parents[1]

    assert _CUDA_SAMPLES_DIR == repository_root / "submodules" / "cuda-samples"


def test_discovers_nested_cmake_and_make_samples(tmp_path):
    (tmp_path / "Samples").mkdir()
    (tmp_path / "Samples" / "CMakeLists.txt").write_text("# aggregate\n")
    category = tmp_path / "Samples" / "0_Introduction"
    category.mkdir()
    (category / "CMakeLists.txt").write_text("# aggregate\n")
    vector_add = _sample(tmp_path, "0_Introduction", "vectorAdd", "CMakeLists.txt")
    clock = _sample(tmp_path, "0_Introduction", "clock", "Makefile")

    discovered = CompatibilityAdapter()._discover_sample_dirs(
        tmp_path / "Samples", None
    )

    assert discovered == [clock, vector_add]


def test_sample_filter_rejects_unknown_names(tmp_path):
    _sample(tmp_path, "0_Introduction", "vectorAdd", "CMakeLists.txt")

    with pytest.raises(ValueError, match="missingSample"):
        CompatibilityAdapter()._discover_sample_dirs(
            tmp_path / "Samples", ["vectorAdd", "missingSample"]
        )


def test_discovery_excludes_nested_cmake_group_manifests(tmp_path):
    group = tmp_path / "Samples" / "8_Platform_Specific" / "Tegra"
    group.mkdir(parents=True)
    (group / "CMakeLists.txt").write_text(
        "add_subdirectory(simpleGLES)\n", encoding="utf-8"
    )
    sample = _sample(
        tmp_path,
        "8_Platform_Specific/Tegra",
        "simpleGLES",
        "CMakeLists.txt",
    )

    discovered = CompatibilityAdapter()._discover_sample_dirs(
        tmp_path / "Samples", None
    )

    assert discovered == [sample]


def test_sample_filter_rejects_an_empty_list(tmp_path):
    _sample(tmp_path, "0_Introduction", "vectorAdd", "CMakeLists.txt")

    with pytest.raises(ValueError, match="non-empty"):
        CompatibilityAdapter()._discover_sample_dirs(tmp_path / "Samples", [])


@pytest.mark.parametrize("sms", ["", '80")\nmessage(FATAL_ERROR injected)'])
def test_architecture_rejects_empty_or_cmake_control_characters(sms):
    with pytest.raises(ValueError, match="invalid architecture"):
        CompatibilityAdapter._validate_architectures(sms)


@pytest.mark.parametrize(
    ("build_system", "manifest", "sms", "compiler", "cmake_command", "jobs"),
    [
        (
            "cmake",
            "CMakeLists.txt",
            "80",
            "/usr/local/cuda/bin/nvcc",
            "/vendor/bin/cmake_maca",
            7,
        ),
        ("make", "Makefile", "70", "/usr/local/musa/bin/mcc", "cmake", 3),
    ],
)
def test_build_commands_include_compiler_arch_and_jobs(
    tmp_path,
    monkeypatch,
    build_system,
    manifest,
    sms,
    compiler,
    cmake_command,
    jobs,
):
    sample_dir = _sample(tmp_path, "0_Introduction", "vectorAdd", manifest)
    build_dir = tmp_path / "build"
    expected_binary = (
        build_dir / "build" if build_system == "cmake" else sample_dir
    ) / "vectorAdd"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        is_build = (len(command) > 1 and command[1] == "--build") or (
            command[0] == "make" and "clean" not in command
        )
        if is_build:
            expected_binary.parent.mkdir(parents=True, exist_ok=True)
            expected_binary.write_text("binary", encoding="utf-8")
            expected_binary.chmod(0o755)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    binary = CompatibilityAdapter()._compile_sample(
        sample_dir,
        build_dir,
        {
            "SMS": sms,
            "CUDACXX": compiler,
            "CMAKE_COMMAND": cmake_command,
        },
        30,
        jobs,
        build_system,
    )

    expected = (
        [
            cmake_command,
            "--build",
            str(build_dir / "build"),
            "--parallel",
            str(jobs),
        ]
        if build_system == "cmake"
        else ["make", f"-j{jobs}", f"SMS={sms}", f"NVCC={compiler}"]
    )
    assert binary == expected_binary
    assert expected in calls


def test_cmake_wrapper_overrides_sample_target_architectures(tmp_path):
    sample_dir = _sample(tmp_path, "0_Introduction", "vectorAdd", "CMakeLists.txt")
    wrapper_dir = tmp_path / "wrapper"

    CompatibilityAdapter._write_cmake_wrapper(wrapper_dir, sample_dir, "ivcore11")

    wrapper = (wrapper_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f'add_subdirectory("{sample_dir.resolve().as_posix()}" sample)' in wrapper
    assert 'PROPERTY CUDA_ARCHITECTURES "ivcore11"' in wrapper


@pytest.mark.parametrize("platform", ["metax", "corex"])
def test_vendor_make_args_remove_nvidia_only_flags(platform):
    args = " ".join(CUDA_SAMPLE_CONFIGS[platform]["make_args"])

    assert all(flag not in args for flag in ("--threads", "-gencode", "-m64"))


def test_corex_make_args_keep_source_language_out_of_link_step():
    compile_args, link_args, _ = CUDA_SAMPLE_CONFIGS["corex"]["make_args"]

    assert "-x ivcore" in compile_args
    assert "-x ivcore" not in link_args


def test_corex_defaults_target_bi_v150():
    config = CUDA_SAMPLE_CONFIGS["corex"]
    compile_args, link_args, _ = config["make_args"]

    assert config["sms"] == "ivcore11"
    assert "--cuda-gpu-arch=ivcore11" in compile_args
    assert "--cuda-gpu-arch=ivcore11" in link_args


def test_compile_env_matches_selected_vendor_toolkit(monkeypatch, compiler_available):
    compiler = "/usr/local/corex/bin/clang++"
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")
    env = CompatibilityAdapter()._build_compile_env("corex", "ivcore11", compiler)

    expected_root = str(Path(compiler).resolve().parent.parent)
    assert env["CUDACXX"] == compiler
    assert env["CUDA_HOME"] == expected_root
    assert env["CUDA_PATH"] == expected_root


@pytest.mark.parametrize(
    ("compiler", "root_parent_index"),
    [
        ("/opt/maca/tools/cu-bridge/bin/cucc", 3),
        ("/opt/maca/mxgpu_llvm/bin/mxcc", 2),
    ],
)
def test_metax_compile_env_infers_maca_path(
    monkeypatch, compiler_available, compiler, root_parent_index
):
    monkeypatch.delenv("MACA_PATH", raising=False)

    env = CompatibilityAdapter()._build_compile_env("metax", "70", compiler)

    assert env["MACA_PATH"] == str(Path(compiler).resolve().parents[root_parent_index])


def test_metax_compile_env_preserves_explicit_maca_path(
    monkeypatch, compiler_available
):
    compiler = "/opt/maca/tools/cu-bridge/bin/cucc"
    monkeypatch.setenv("MACA_PATH", "/custom/maca")
    env = CompatibilityAdapter()._build_compile_env("metax", "70", compiler)

    assert env["MACA_PATH"] == "/custom/maca"


def test_metax_compile_env_selects_cu_bridge_cmake(monkeypatch, compiler_available):
    compiler = "/opt/maca/tools/cu-bridge/bin/cucc"

    env = CompatibilityAdapter()._build_compile_env("metax", "70", compiler)

    cmake_command = "/opt/maca/tools/cu-bridge/tools/cmake_maca"
    assert env["CMAKE_COMMAND"] == cmake_command
    assert env["CUCC_CMAKE_ENTRY"] == "2"
    assert env["PATH"].split(os.pathsep)[0] == str(Path(cmake_command).resolve().parent)


def test_cuda_sample_metrics_keep_skips_separate(tmp_path, monkeypatch):
    samples = [
        _sample(tmp_path, "0_Introduction", name, "CMakeLists.txt")
        for name in ("passes", "fails", "waived")
    ]
    binaries = {sample.name: tmp_path / f"{sample.name}.bin" for sample in samples}
    adapter = CompatibilityAdapter()

    monkeypatch.setattr(adapter, "_discover_sample_dirs", lambda *_: samples)
    monkeypatch.setattr(
        adapter,
        "_build_compile_env",
        lambda *_: {"SMS": "80", "CUDACXX": "/usr/local/cuda/bin/nvcc"},
    )
    monkeypatch.setattr(
        adapter,
        "_compile_sample",
        lambda sample_dir, *_: binaries[sample_dir.name],
    )
    outcomes = {
        "passes.bin": ("pass", ""),
        "fails.bin": ("fail", "kernel failed"),
        "waived.bin": ("skip", "sample waived"),
    }
    monkeypatch.setattr(
        adapter, "_run_sample", lambda binary, *_: outcomes[binary.name]
    )

    config = {"platform": "nvidia", "cuda_samples_dir": str(tmp_path)}
    metrics = adapter._run_cuda_samples_test(config)
    values = {metric["name"]: metric["value"] for metric in metrics}

    assert config["compiler"] == "/usr/local/cuda/bin/nvcc"
    assert config["sms"] == "80"
    assert values["compatibility.cuda_samples.run_passed"] == 1
    assert values["compatibility.cuda_samples.run_failed"] == 1
    assert values["compatibility.cuda_samples.run_skipped"] == 1
    assert values["compatibility.cuda_samples.run_pass_rate"] == 33.33


@pytest.mark.parametrize("failure_stage", ["compile", "run"])
def test_sample_failures_are_accounted_for(tmp_path, monkeypatch, failure_stage):
    sample = _sample(tmp_path, "0_Introduction", "vectorAdd", "CMakeLists.txt")
    adapter = CompatibilityAdapter()
    monkeypatch.setattr(adapter, "_discover_sample_dirs", lambda *_: [sample])
    monkeypatch.setattr(
        adapter,
        "_build_compile_env",
        lambda *_: {"SMS": "80", "CUDACXX": "/usr/local/cuda/bin/nvcc"},
    )

    def compile_sample(*_):
        if failure_stage == "compile":
            raise RuntimeError("compiler failed")
        return tmp_path / "vectorAdd"

    def run_sample(*_):
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(adapter, "_compile_sample", compile_sample)
    monkeypatch.setattr(adapter, "_run_sample", run_sample)

    metrics = adapter._run_cuda_samples_test(
        {"platform": "nvidia", "cuda_samples_dir": str(tmp_path)}
    )
    values = {metric["name"]: metric["value"] for metric in metrics}
    details = values["compatibility.cuda_samples.details"]

    compile_passed = int(failure_stage == "run")
    assert values["compatibility.cuda_samples.compile_passed"] == compile_passed
    assert values["compatibility.cuda_samples.run_failed"] == compile_passed
    assert values["compatibility.cuda_samples.run_skipped"] == 0
    assert details[0]["compile_result"] == ("pass" if compile_passed else "fail")
    assert details[0]["run_result"] == ("fail" if compile_passed else "not_run")


def test_run_sample_classifies_cuda_waiver(tmp_path):
    binary = tmp_path / "sample"
    binary.write_text("binary", encoding="utf-8")
    binary.chmod(0o755)
    adapter = CompatibilityAdapter()

    result, error = adapter._classify_run_result(
        subprocess.CompletedProcess([str(binary)], 2, "Sample waived", "")
    )

    assert result == "skip"
    assert "waived" in error.lower()


def test_run_sample_uses_the_binary_directory(tmp_path, monkeypatch):
    binary = tmp_path / "build" / "sample"
    binary.parent.mkdir()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "Result = PASS", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result, _ = CompatibilityAdapter()._run_sample(binary, 30)

    assert result == "pass"
    assert calls == [
        (
            [str(binary)],
            {
                "cwd": str(binary.parent),
                "capture_output": True,
                "text": True,
                "errors": "replace",
                "env": None,
                "timeout": 30,
            },
        )
    ]


def test_run_sample_resolves_a_relative_binary_path(tmp_path, monkeypatch):
    binary = Path("build") / "sample"
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, "Result = PASS", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result, _ = CompatibilityAdapter()._run_sample(binary, 30)

    resolved_binary = binary.resolve()
    assert result == "pass"
    assert calls == [([str(resolved_binary)], str(resolved_binary.parent))]


def test_metrics_reject_an_empty_sample_set():
    with pytest.raises(ValueError, match="No CUDA samples selected"):
        CompatibilityAdapter._build_metrics([])


def test_dispatcher_only_registers_cuda_samples_compatibility():
    assert isinstance(
        Dispatcher()._create_adapter("compatibility", "cudasamples"),
        CompatibilityAdapter,
    )
    for framework in ("megatron", "vllm", "infinilm"):
        with pytest.raises(ValueError, match="Adapter not registered"):
            Dispatcher()._create_adapter("compatibility", framework)

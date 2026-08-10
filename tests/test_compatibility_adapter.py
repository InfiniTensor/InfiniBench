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


def _sample(
    root: Path,
    name: str,
    manifest: str = "CMakeLists.txt",
    category: str = "0_Introduction",
) -> Path:
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


def _mock_successful_build(monkeypatch, binary):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "--build" in command or (command[0] == "make" and "clean" not in command):
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("binary", encoding="utf-8")
            binary.chmod(0o755)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_default_cuda_samples_dir_uses_repository_submodule():
    repository_root = Path(__file__).resolve().parents[1]

    assert _CUDA_SAMPLES_DIR == repository_root / "submodules" / "cuda-samples"


def test_discovers_only_standalone_cmake_and_make_samples(tmp_path):
    (tmp_path / "Samples").mkdir()
    (tmp_path / "Samples" / "CMakeLists.txt").write_text("# aggregate\n")
    category = tmp_path / "Samples" / "0_Introduction"
    category.mkdir()
    (category / "CMakeLists.txt").write_text("# aggregate\n")
    vector_add = _sample(tmp_path, "vectorAdd")
    clock = _sample(tmp_path, "clock", "Makefile")

    group = tmp_path / "Samples" / "8_Platform_Specific" / "Tegra"
    group.mkdir(parents=True)
    (group / "CMakeLists.txt").write_text(
        "add_subdirectory(simpleGLES)\n", encoding="utf-8"
    )
    simple_gles = _sample(
        tmp_path,
        "simpleGLES",
        category="8_Platform_Specific/Tegra",
    )

    discovered = CompatibilityAdapter()._discover_sample_dirs(
        tmp_path / "Samples", None
    )

    assert discovered == [clock, vector_add, simple_gles]


@pytest.mark.parametrize(
    ("sample_filter", "error"),
    [([], "non-empty"), (["vectorAdd", "missingSample"], "missingSample")],
)
def test_sample_filter_rejects_invalid_lists(tmp_path, sample_filter, error):
    _sample(tmp_path, "vectorAdd")

    with pytest.raises(ValueError, match=error):
        CompatibilityAdapter()._discover_sample_dirs(
            tmp_path / "Samples", sample_filter
        )


@pytest.mark.parametrize("sms", ["", '80")\nmessage(FATAL_ERROR injected)'])
def test_architecture_rejects_empty_or_cmake_control_characters(sms):
    with pytest.raises(ValueError, match="invalid architecture"):
        CompatibilityAdapter._validate_architectures(sms)


def test_cmake_build_includes_compiler_arch_and_jobs(tmp_path, monkeypatch):
    sample_dir = _sample(tmp_path, "vectorAdd")
    build_dir = tmp_path / "build"
    compiler = "/usr/local/cuda/bin/nvcc"
    cmake_command = "/usr/bin/cmake"
    sms, jobs = "80", 4
    expected_binary = build_dir / "build" / "vectorAdd"
    calls = _mock_successful_build(monkeypatch, expected_binary)
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
        "cmake",
    )

    assert binary == expected_binary
    assert f"-DCMAKE_CUDA_ARCHITECTURES={sms}" in calls[0][0]
    assert f"-DCMAKE_CUDA_COMPILER={compiler}" in calls[0][0]
    assert calls[1][0][-2:] == ["--parallel", str(jobs)]


def test_cmake_wrapper_overrides_sample_target_architectures(tmp_path):
    sample_dir = _sample(tmp_path, "vectorAdd")
    wrapper_dir = tmp_path / "wrapper"

    CompatibilityAdapter._write_cmake_wrapper(wrapper_dir, sample_dir, "ivcore11")

    wrapper = (wrapper_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    assert f'add_subdirectory("{sample_dir.resolve().as_posix()}" sample)' in wrapper
    assert 'PROPERTY CUDA_ARCHITECTURES "ivcore11"' in wrapper


def test_metax_cmake_wrapper_owns_toolchain_detection(tmp_path, monkeypatch):
    sample_dir = _sample(tmp_path, "vectorAdd")
    build_dir = tmp_path / "build"
    binary = build_dir / "build" / "vectorAdd"
    calls = _mock_successful_build(monkeypatch, binary)
    cmake_command = "/opt/maca/tools/cu-bridge/tools/cmake_maca"

    result = CompatibilityAdapter()._compile_sample(
        sample_dir,
        build_dir,
        {
            "SMS": "70",
            "CUDACXX": "/opt/maca/tools/cu-bridge/bin/cucc",
            "CMAKE_COMMAND": cmake_command,
        },
        30,
        4,
        "cmake",
    )

    configure_command, configure_kwargs = calls[0]
    configure_env = configure_kwargs["env"]
    assert result == binary
    assert not any(arg.startswith("-DCMAKE_CUDA_") for arg in configure_command)
    assert "CUDACXX" not in configure_env
    assert configure_env["WCUDA_HOME"] == str(build_dir / "cmake-maca")
    assert calls[1][1]["env"]["WCUDA_HOME"] == configure_env["WCUDA_HOME"]


def test_corex_defaults_target_bi_v150():
    config = CUDA_SAMPLE_CONFIGS["corex"]
    compile_args, link_args, _ = config["make_args"]

    assert config["sms"] == "ivcore11"
    assert "--cuda-gpu-arch=ivcore11" in compile_args
    assert "--cuda-gpu-arch=ivcore11" in link_args


def test_metax_compile_env_selects_wrapper(monkeypatch, compiler_available):
    compiler = "/opt/maca/tools/cu-bridge/bin/cucc"
    monkeypatch.delenv("MACA_PATH", raising=False)

    env = CompatibilityAdapter()._build_compile_env("metax", "70", compiler)

    cmake_command = "/opt/maca/tools/cu-bridge/tools/cmake_maca"
    assert env["MACA_PATH"] == str(Path(compiler).resolve().parents[3])
    assert env["CMAKE_COMMAND"] == cmake_command
    assert env["CUCC_CMAKE_ENTRY"] == "2"
    assert env["PATH"].split(os.pathsep)[0] == str(Path(cmake_command).resolve().parent)


def test_cuda_sample_metrics_keep_skips_separate(tmp_path, monkeypatch):
    samples = [_sample(tmp_path, name) for name in ("passes", "fails", "waived")]
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


def test_compile_failures_are_accounted_for(tmp_path, monkeypatch):
    sample = _sample(tmp_path, "vectorAdd")
    adapter = CompatibilityAdapter()
    monkeypatch.setattr(adapter, "_discover_sample_dirs", lambda *_: [sample])
    monkeypatch.setattr(
        adapter,
        "_build_compile_env",
        lambda *_: {"SMS": "80", "CUDACXX": "/usr/local/cuda/bin/nvcc"},
    )

    def compile_sample(*_):
        raise RuntimeError("compiler failed")

    monkeypatch.setattr(adapter, "_compile_sample", compile_sample)

    metrics = adapter._run_cuda_samples_test(
        {"platform": "nvidia", "cuda_samples_dir": str(tmp_path)}
    )
    values = {metric["name"]: metric["value"] for metric in metrics}
    details = values["compatibility.cuda_samples.details"]

    assert values["compatibility.cuda_samples.compile_passed"] == 0
    assert values["compatibility.cuda_samples.run_failed"] == 0
    assert values["compatibility.cuda_samples.run_skipped"] == 0
    assert details[0]["compile_result"] == "fail"
    assert details[0]["run_result"] == "not_run"


def test_run_sample_classifies_cuda_waiver():
    result, error = CompatibilityAdapter._classify_run_result(
        subprocess.CompletedProcess(["sample"], 2, "Sample waived", "")
    )

    assert result == "skip"
    assert "waived" in error.lower()


def test_run_sample_resolves_path_and_uses_binary_directory(tmp_path, monkeypatch):
    binary = Path("build") / "sample"
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "Result = PASS", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result, _ = CompatibilityAdapter()._run_sample(binary, 30)

    resolved_binary = binary.resolve()
    assert result == "pass"
    assert calls == [
        (
            [str(resolved_binary)],
            {
                "cwd": str(resolved_binary.parent),
                "capture_output": True,
                "text": True,
                "errors": "replace",
                "env": None,
                "timeout": 30,
            },
        )
    ]


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

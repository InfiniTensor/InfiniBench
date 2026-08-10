#!/usr/bin/env python3
"""CUDA Samples compatibility test adapter.

Compiles and runs CUDA Samples on supported CUDA-compatible platforms and
reports compile/run pass rates.

Testcase format:
    compatibility.CudaSamples.PassRate
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from infinibench.adapter import BaseAdapter
from infinibench.common.constants import CUDA_SAMPLE_CONFIGS, ErrorCode, InfiniBenchJson
from infinibench.hardware.constants import PLATFORM_ALIASES, PLATFORM_CONFIGS
from infinibench.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)
_METRIC_PREFIX = "compatibility.cuda_samples"

# Repo root for finding cuda-samples
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDA_SAMPLES_DIR = (
    _REPO_ROOT / "InfiniPerf" / "benchmarks" / "compatibility" / "cuda-samples"
)


class CompatibilityAdapter(BaseAdapter):
    """Adapter for CUDA Samples compatibility tests."""

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process compatibility test."""
        test_dict = self._normalize_test_input(test_input)
        if not test_dict:
            return self._create_error_response(
                "Invalid test input format", result_code=ErrorCode.CONFIG
            )

        testcase = test_dict.get(InfiniBenchJson.TESTCASE, "unknown")
        config = test_dict.get(InfiniBenchJson.CONFIG, {})
        run_id = test_dict.get(InfiniBenchJson.RUN_ID, "unknown")

        logger.info(f"CompatibilityAdapter: Processing {testcase}")

        # Extract sub-test type from testcase (third component)
        parts = testcase.split(".")
        if len(parts) < 3:
            return self._create_error_response(
                f"Invalid testcase format: {testcase}. "
                f"Expected: compatibility.<SubTest>.<Detail>",
                test_dict,
                result_code=ErrorCode.CONFIG,
            )

        if parts[1].lower() != "cudasamples":
            return self._create_error_response(
                f"Unknown compatibility sub-test: {parts[1].lower()}",
                test_dict,
                result_code=ErrorCode.CONFIG,
            )

        return {
            InfiniBenchJson.RESULT_CODE: 0,
            InfiniBenchJson.TIME: get_timestamp(),
            InfiniBenchJson.RUN_ID: run_id,
            InfiniBenchJson.TESTCASE: testcase,
            InfiniBenchJson.CONFIG: config,
            InfiniBenchJson.METRICS: self._run_cuda_samples_test(config),
        }

    # ------------------------------------------------------------------
    # cuda-samples
    # ------------------------------------------------------------------

    def _run_cuda_samples_test(self, config: Dict[str, Any]) -> List[Dict]:
        """Compile and run cuda-samples, collect pass rate."""
        requested_platform = str(config.get("platform", "cuda")).lower().strip()
        platform = PLATFORM_ALIASES.get(requested_platform)
        if not platform:
            raise ValueError(
                f"Unsupported CUDA-compatible platform: {requested_platform}"
            )

        samples_dir = config.get("cuda_samples_dir", str(_CUDA_SAMPLES_DIR))
        platform_config = CUDA_SAMPLE_CONFIGS.get(platform)
        if platform_config is None:
            raise KeyError(
                f"CUDA Samples configuration not found for platform: {platform}"
            )
        sms = self._validate_architectures(config.get("sms", platform_config["sms"]))
        timeout_per_sample = self._positive_int(
            config.get("timeout_per_sample", 60), "timeout_per_sample"
        )
        jobs = self._positive_int(config.get("jobs", os.cpu_count() or 1), "jobs")
        sample_filter = config.get("sample_filter")
        build_system = str(config.get("build_system", "auto")).lower()
        make_args = config.get("make_args")
        if make_args is None:
            make_args = list(platform_config["make_args"])
        elif not isinstance(make_args, list) or not all(
            isinstance(argument, str) for argument in make_args
        ):
            raise ValueError("make_args must be a list of strings")

        samples_root = Path(samples_dir) / "Samples"
        if not samples_root.exists():
            raise FileNotFoundError(f"CUDA samples directory not found: {samples_root}")

        sample_dirs = self._discover_sample_dirs(samples_root, sample_filter)
        env = self._build_compile_env(platform, sms, config.get("compiler"))
        config.setdefault("compiler", env["CUDACXX"])
        config.setdefault("sms", sms)

        details = [
            self._test_sample(
                sample_dir,
                samples_root,
                env,
                timeout_per_sample,
                jobs,
                build_system,
                make_args,
            )
            for sample_dir in sample_dirs
        ]
        return self._build_metrics(details)

    def _test_sample(
        self,
        sample_dir: Path,
        samples_root: Path,
        env: Dict[str, str],
        timeout: int,
        jobs: int,
        build_system: str,
        make_args: List[str],
    ) -> Dict[str, Any]:
        selected_build_system = self._select_build_system(sample_dir, build_system)
        compile_result, run_result, error = "fail", "not_run", ""

        with tempfile.TemporaryDirectory(
            prefix=f"infinibench-{sample_dir.name}-"
        ) as temp_dir:
            try:
                binary = self._compile_sample(
                    sample_dir,
                    Path(temp_dir),
                    env,
                    timeout,
                    jobs,
                    selected_build_system,
                    make_args,
                )
                compile_result = "pass"
                run_result, error = self._run_sample(binary, timeout, env)
            except subprocess.TimeoutExpired:
                error = f"Compilation timed out after {timeout} seconds"
            except Exception as exc:
                if compile_result == "pass":
                    run_result = "fail"
                error = str(exc)
            finally:
                if selected_build_system == "make":
                    self._clean_make_sample(sample_dir, env, timeout)

        return {
            "name": sample_dir.name,
            "path": sample_dir.relative_to(samples_root).as_posix(),
            "compile_result": compile_result,
            "run_result": run_result,
            "error": error[-2000:] if error else "",
        }

    @staticmethod
    def _build_metrics(details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        total = len(details)
        if total == 0:
            raise ValueError("No CUDA samples selected")
        compile_passed = sum(d["compile_result"] == "pass" for d in details)
        run_passed = sum(d["run_result"] == "pass" for d in details)
        values = {
            "total": total,
            "compile_passed": compile_passed,
            "compile_failed": total - compile_passed,
            "compile_pass_rate": round(compile_passed / total * 100, 2),
            "run_passed": run_passed,
            "run_failed": sum(d["run_result"] == "fail" for d in details),
            "run_skipped": sum(d["run_result"] == "skip" for d in details),
            "run_pass_rate": round(run_passed / total * 100, 2),
        }
        metrics = [
            {
                "name": f"{_METRIC_PREFIX}.{name}",
                "value": value,
                "type": "scalar",
                "unit": "%" if name.endswith("pass_rate") else "",
            }
            for name, value in values.items()
        ]
        metrics.append(
            {
                "name": f"{_METRIC_PREFIX}.details",
                "value": details,
                "type": "detail",
                "unit": "",
            }
        )
        return metrics

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    @staticmethod
    def _validate_architectures(value: Any) -> str:
        architectures = str(value).strip()
        if not architectures or not re.fullmatch(r"[A-Za-z0-9_.+; -]+", architectures):
            raise ValueError("sms contains invalid architecture characters")
        return architectures

    @staticmethod
    def _discover_sample_dirs(
        samples_root: Path, sample_filter: Optional[List[str]]
    ) -> List[Path]:
        sample_dirs = sorted(
            {
                manifest.parent
                for manifest_name in ("CMakeLists.txt", "Makefile")
                for manifest in samples_root.rglob(manifest_name)
                if len(manifest.parent.relative_to(samples_root).parts) >= 2
                and CompatibilityAdapter._is_standalone_manifest(manifest)
            }
        )
        if not sample_dirs:
            raise ValueError(f"No CUDA samples found under: {samples_root}")
        if sample_filter is None:
            return sample_dirs
        if (
            not isinstance(sample_filter, list)
            or not sample_filter
            or not all(isinstance(name, str) and name for name in sample_filter)
        ):
            raise ValueError("sample_filter must be a non-empty list of names")

        requested = set(sample_filter)
        selected = [sample for sample in sample_dirs if sample.name in requested]
        found = {sample.name for sample in selected}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"CUDA sample filters not found: {', '.join(missing)}")
        return selected

    @staticmethod
    def _is_standalone_manifest(manifest: Path) -> bool:
        if manifest.name == "Makefile":
            return True
        try:
            content = manifest.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return (
            re.search(r"^\s*project\s*\(", content, re.IGNORECASE | re.MULTILINE)
            is not None
        )

    @staticmethod
    def _select_build_system(sample_dir: Path, requested: str) -> str:
        if requested not in {"auto", "cmake", "make"}:
            raise ValueError("build_system must be one of: auto, cmake, make")
        if requested == "auto":
            if (sample_dir / "CMakeLists.txt").exists():
                return "cmake"
            if (sample_dir / "Makefile").exists():
                return "make"
            raise ValueError(f"No supported build manifest in: {sample_dir}")

        manifest = "CMakeLists.txt" if requested == "cmake" else "Makefile"
        if not (sample_dir / manifest).exists():
            raise ValueError(f"{manifest} not found in CUDA sample: {sample_dir}")
        return requested

    def _build_compile_env(
        self, platform: str, sms: str, compiler: Optional[str] = None
    ) -> Dict[str, str]:
        """Build environment variables for compilation."""
        env = os.environ.copy()
        env["SMS"] = sms

        candidates = (
            (compiler,) if compiler else PLATFORM_CONFIGS[platform]["compilers"]
        )
        compiler_path = next(
            (
                resolved
                for candidate in candidates
                if (resolved := shutil.which(candidate, path=env.get("PATH")))
            ),
            None,
        )
        if not compiler_path:
            raise FileNotFoundError(
                f"No compiler found for platform {platform}; checked: "
                + ", ".join(candidates)
            )

        env["CUDACXX"] = compiler_path
        toolkit_root = str(Path(compiler_path).resolve().parent.parent)
        env["CUDA_HOME"] = toolkit_root
        env["CUDA_PATH"] = toolkit_root
        if platform == "metax" and not env.get("MACA_PATH"):
            maca_path = self._infer_metax_root(compiler_path)
            if maca_path:
                env["MACA_PATH"] = maca_path

        return env

    @staticmethod
    def _infer_metax_root(compiler_path: str) -> Optional[str]:
        path = Path(compiler_path).resolve()
        if path.parts[-4:] == ("tools", "cu-bridge", "bin", "cucc"):
            return str(path.parents[3])
        if path.parts[-3:] == ("mxgpu_llvm", "bin", "mxcc"):
            return str(path.parents[2])
        return None

    def _compile_sample(
        self,
        sample_dir: Path,
        build_dir: Path,
        env: Dict[str, str],
        timeout: int,
        jobs: int,
        build_system: str,
        make_args: Optional[List[str]] = None,
    ) -> Path:
        """Compile one CUDA sample and return its executable."""
        sms = env.get("SMS", "80")
        if build_system == "cmake":
            wrapper_dir = build_dir / "source"
            cmake_build_dir = build_dir / "build"
            self._write_cmake_wrapper(wrapper_dir, sample_dir, sms)
            cmake_build_dir.mkdir(parents=True, exist_ok=True)
            configure_result = self._run_build_command(
                [
                    "cmake",
                    "-S",
                    str(wrapper_dir),
                    "-B",
                    str(cmake_build_dir),
                    f"-DCMAKE_CUDA_ARCHITECTURES={sms}",
                    f"-DCMAKE_CUDA_COMPILER={env['CUDACXX']}",
                ],
                sample_dir,
                env,
                timeout,
            )
            if configure_result.returncode:
                raise RuntimeError(self._command_error(configure_result))
            command = [
                "cmake",
                "--build",
                str(cmake_build_dir),
                "--parallel",
                str(jobs),
            ]
            search_root = cmake_build_dir
        else:
            self._run_build_command(["make", "clean"], sample_dir, env, timeout)
            command = [
                "make",
                f"-j{jobs}",
                f"SMS={sms}",
                f"NVCC={env['CUDACXX']}",
                *(make_args or []),
            ]
            search_root = sample_dir

        build_result = self._run_build_command(command, sample_dir, env, timeout)
        if build_result.returncode:
            raise RuntimeError(self._command_error(build_result))

        binary = self._find_sample_binary(search_root, sample_dir.name)
        if not binary:
            raise RuntimeError(
                f"Build succeeded but no executable was found for {sample_dir.name}"
            )
        return binary

    @staticmethod
    def _write_cmake_wrapper(wrapper_dir: Path, sample_dir: Path, sms: str) -> None:
        """Create an out-of-tree wrapper that owns the target architecture."""
        wrapper_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir.resolve().as_posix().replace('"', '\\"')
        architecture = sms.replace('"', '\\"')
        content = f"""cmake_minimum_required(VERSION 3.20)
project(InfiniBenchCudaSample LANGUAGES C CXX CUDA)

add_subdirectory("{sample_path}" sample)

function(infinibench_set_cuda_architectures directory)
  get_property(targets DIRECTORY "${{directory}}" PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(target_type "${{target}}" TYPE)
    if(NOT target_type STREQUAL "UTILITY" AND
       NOT target_type STREQUAL "INTERFACE_LIBRARY")
      set_property(TARGET "${{target}}" PROPERTY CUDA_ARCHITECTURES "{architecture}")
    endif()
  endforeach()
  get_property(subdirectories DIRECTORY "${{directory}}" PROPERTY SUBDIRECTORIES)
  foreach(subdirectory IN LISTS subdirectories)
    infinibench_set_cuda_architectures("${{subdirectory}}")
  endforeach()
endfunction()

infinibench_set_cuda_architectures("{sample_path}")
"""
        (wrapper_dir / "CMakeLists.txt").write_text(content, encoding="utf-8")

    @staticmethod
    def _run_build_command(
        command: List[str], cwd: Path, env: Dict[str, str], timeout: int
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            env=env,
            timeout=timeout,
        )

    @staticmethod
    def _command_error(result: subprocess.CompletedProcess) -> str:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        return output[-2000:] or f"Command exited with code {result.returncode}"

    @staticmethod
    def _find_sample_binary(search_root: Path, sample_name: str) -> Optional[Path]:
        for name in (sample_name, sample_name.replace("_", "")):
            direct = search_root / name
            if direct.is_file() and os.access(direct, os.X_OK):
                return direct
            matches = sorted(
                path
                for path in search_root.rglob(name)
                if path.is_file() and os.access(path, os.X_OK)
            )
            if matches:
                return matches[0]
        return None

    def _run_sample(
        self, binary: Path, timeout: int, env: Optional[Dict[str, str]] = None
    ) -> Tuple[str, str]:
        """Run a compiled CUDA sample."""
        binary = binary.resolve()
        try:
            result = subprocess.run(
                [str(binary)],
                cwd=str(binary.parent),
                capture_output=True,
                text=True,
                errors="replace",
                env=env,
                timeout=timeout,
            )
            return self._classify_run_result(result)
        except subprocess.TimeoutExpired:
            return "fail", f"Execution timed out after {timeout} seconds"
        except FileNotFoundError as exc:
            return "fail", str(exc)

    @staticmethod
    def _classify_run_result(
        result: subprocess.CompletedProcess,
    ) -> Tuple[str, str]:
        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        normalized = output.lower()
        if result.returncode == 2 or any(
            marker in normalized
            for marker in ("sample waived", "waiving sample", "result = waived")
        ):
            return "skip", output[-2000:] or "Sample waived"
        if result.returncode == 0 and not re.search(r"result\s*=\s*fail", normalized):
            return "pass", ""
        return (
            "fail",
            output[-2000:] or f"Executable exited with code {result.returncode}",
        )

    def _clean_make_sample(
        self, sample_dir: Path, env: Dict[str, str], timeout: int
    ) -> None:
        try:
            self._run_build_command(["make", "clean"], sample_dir, env, timeout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("Failed to clean CUDA sample build: %s", sample_dir)

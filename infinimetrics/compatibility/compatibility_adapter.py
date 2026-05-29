#!/usr/bin/env python3
"""Compatibility Test Adapter.

Tests CUDA compatibility (cuda-samples compile/run pass rate) and
framework compatibility (Megatron/vLLM/InfiniLM startup) on CUDA-compatible
platforms (MetaX, Iluvatar, Hygon, Moore).

Testcase format:
    compatibility.CudaSamples.PassRate
    compatibility.Megatron.Startup
    compatibility.VLLM.Startup
    compatibility.InfiniLM.Startup
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.constants import InfiniMetricsJson, ErrorCode
from infinimetrics.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)

# Repo root for finding cuda-samples
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CUDA_SAMPLES_DIR = (
    _REPO_ROOT
    / "InfiniPerf"
    / "benchmarks"
    / "compatibility"
    / "cuda-samples"
)

# Platform → (SMS / CUDA_COMPUTE_CAPABILITY, extra env)
PLATFORM_COMPILE_CONFIG = {
    "metax": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "metax_gpu": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "iluvatar": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "iluvatar_gpu": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "hygon": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "sugon_dcu": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "moore": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "moore_gpu": {"SMS": "70", "CUDA_PATH_ENV": "CUDA_HOME"},
    "nvidia": {"SMS": "80", "CUDA_PATH_ENV": "CUDA_HOME"},
    "cuda": {"SMS": "80", "CUDA_PATH_ENV": "CUDA_HOME"},
}


class CompatibilityAdapter(BaseAdapter):
    """Adapter for CUDA compatibility and framework startup tests."""

    def __init__(self):
        self.config = {}

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources."""
        self.config = config

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Process compatibility test."""
        test_dict = self._normalize_test_input(test_input)
        if not test_dict:
            return self._create_error_response(
                "Invalid test input format", result_code=ErrorCode.CONFIG
            )

        testcase = test_dict.get(InfiniMetricsJson.TESTCASE, "unknown")
        config = test_dict.get(InfiniMetricsJson.CONFIG, {})
        run_id = test_dict.get(InfiniMetricsJson.RUN_ID, "unknown")

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

        sub_test = parts[1].lower()

        try:
            if sub_test == "cudasamples":
                metrics = self._run_cuda_samples_test(config)
            elif sub_test == "megatron":
                metrics = self._run_framework_startup("megatron", config)
            elif sub_test == "vllm":
                metrics = self._run_framework_startup("vllm", config)
            elif sub_test == "infinilm":
                metrics = self._run_framework_startup("infinilm", config)
            else:
                return self._create_error_response(
                    f"Unknown compatibility sub-test: {sub_test}",
                    test_dict,
                    result_code=ErrorCode.CONFIG,
                )

            return {
                InfiniMetricsJson.RESULT_CODE: 0,
                InfiniMetricsJson.TIME: get_timestamp(),
                InfiniMetricsJson.RUN_ID: run_id,
                InfiniMetricsJson.TESTCASE: testcase,
                InfiniMetricsJson.CONFIG: config,
                InfiniMetricsJson.METRICS: metrics,
            }

        except Exception as e:
            logger.error(
                f"CompatibilityAdapter: Test failed for {testcase}: {e}",
                exc_info=True,
            )
            raise

    # ------------------------------------------------------------------
    # cuda-samples
    # ------------------------------------------------------------------

    def _run_cuda_samples_test(self, config: Dict[str, Any]) -> List[Dict]:
        """Compile and run cuda-samples, collect pass rate."""
        platform = config.get("platform", "nvidia").lower()
        samples_dir = config.get("cuda_samples_dir", str(_CUDA_SAMPLES_DIR))
        sms = config.get("sms", PLATFORM_COMPILE_CONFIG.get(platform, {}).get("SMS", "80"))
        timeout_per_sample = config.get("timeout_per_sample", 60)
        sample_filter = config.get("sample_filter", None)  # list of sample names to test

        samples_root = Path(samples_dir) / "Samples"
        if not samples_root.exists():
            raise FileNotFoundError(f"CUDA samples directory not found: {samples_root}")

        # Discover individual sample directories (two levels deep:
        # Samples/<category>/<sample>/)
        sample_dirs = sorted(
            d
            for cat_dir in samples_root.iterdir()
            if cat_dir.is_dir() and (cat_dir / "CMakeLists.txt").exists()
            for d in cat_dir.iterdir()
            if d.is_dir() and (d / "CMakeLists.txt").exists()
        )

        if sample_filter:
            sample_dirs = [
                d for d in sample_dirs if d.name in sample_filter
            ]

        total = len(sample_dirs)
        if total == 0:
            return [{"name": "compatibility.cuda_samples.total", "value": 0, "type": "scalar", "unit": ""}]

        compile_passed = 0
        run_passed = 0
        details = []

        env = self._build_compile_env(platform, sms)

        for sample_dir in sample_dirs:
            name = sample_dir.name
            compile_result = "fail"
            run_result = "skip"
            error = ""

            # Compile
            try:
                compile_result, error = self._compile_sample(
                    sample_dir, env, timeout_per_sample
                )
                if compile_result == "pass":
                    compile_passed += 1
            except Exception as e:
                error = str(e)

            # Run (only if compile succeeded)
            if compile_result == "pass":
                try:
                    run_result, run_error = self._run_sample(
                        sample_dir, timeout_per_sample
                    )
                    if run_result == "pass":
                        run_passed += 1
                    else:
                        error = run_error
                except Exception as e:
                    run_result = "fail"
                    error = str(e)

            details.append(
                {
                    "name": name,
                    "compile_result": compile_result,
                    "run_result": run_result,
                    "error": error[:500] if error else "",
                }
            )

        metrics = [
            {
                "name": "compatibility.cuda_samples.total",
                "value": total,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "compatibility.cuda_samples.compile_passed",
                "value": compile_passed,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "compatibility.cuda_samples.compile_failed",
                "value": total - compile_passed,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "compatibility.cuda_samples.compile_pass_rate",
                "value": round(compile_passed / total * 100, 2) if total > 0 else 0,
                "type": "scalar",
                "unit": "%",
            },
            {
                "name": "compatibility.cuda_samples.run_passed",
                "value": run_passed,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "compatibility.cuda_samples.run_failed",
                "value": compile_passed - run_passed,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "compatibility.cuda_samples.run_pass_rate",
                "value": round(run_passed / total * 100, 2) if total > 0 else 0,
                "type": "scalar",
                "unit": "%",
            },
            {
                "name": "compatibility.cuda_samples.details",
                "value": details,
                "type": "detail",
                "unit": "",
            },
        ]

        return metrics

    def _build_compile_env(self, platform: str, sms: str) -> Dict[str, str]:
        """Build environment variables for compilation."""
        env = os.environ.copy()
        env["SMS"] = sms

        plat_config = PLATFORM_COMPILE_CONFIG.get(platform, {})
        cuda_path_env = plat_config.get("CUDA_PATH_ENV", "CUDA_HOME")
        if cuda_path_env not in env or not env[cuda_path_env]:
            # Try to find CUDA automatically
            for candidate in ["/usr/local/cuda", "/usr/local/cuda-12"]:
                if Path(candidate).exists():
                    env[cuda_path_env] = candidate
                    break

        return env

    def _compile_sample(
        self, sample_dir: Path, env: Dict[str, str], timeout: int
    ) -> tuple:
        """Compile a single cuda sample. Returns (result, error)."""
        build_dir = sample_dir / "build"
        build_dir.mkdir(exist_ok=True)

        sms = env.get("SMS", "80")
        cmake_lists = sample_dir / "CMakeLists.txt"

        # Patch hardcoded CMAKE_CUDA_ARCHITECTURES in CMakeLists.txt
        # cuda-samples uses set() which overrides -D cache variables
        if cmake_lists.exists():
            subprocess.run(
                [
                    "sed", "-i",
                    f"s/set(CMAKE_CUDA_ARCHITECTURES.*/set(CMAKE_CUDA_ARCHITECTURES {sms})/",
                    str(cmake_lists),
                ],
                capture_output=True,
                text=True,
            )

        # cmake
        cmake_result = subprocess.run(
            [
                "cmake", "..",
                f"-DSMS={sms}",
                f"-DCMAKE_CUDA_ARCHITECTURES={sms}",
            ],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if cmake_result.returncode != 0:
            return "fail", cmake_result.stderr[-500:]

        # make
        nproc = os.cpu_count() or 4
        make_result = subprocess.run(
            ["make", f"-j{nproc}"],
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        if make_result.returncode != 0:
            return "fail", make_result.stderr[-500:]

        return "pass", ""

    def _run_sample(self, sample_dir: Path, timeout: int) -> tuple:
        """Run a compiled cuda sample. Returns (result, error)."""
        # Find the compiled binary
        build_dir = sample_dir / "build"
        binaries = list(build_dir.glob(sample_dir.name.replace("_", "")))
        if not binaries:
            binaries = list(build_dir.glob("*"))
            binaries = [
                b
                for b in binaries
                if b.is_file() and os.access(b, os.X_OK) and not b.name.startswith(".")
            ]

        if not binaries:
            return "skip", "No binary found after compilation"

        binary = binaries[0]
        try:
            result = subprocess.run(
                [str(binary)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                return "pass", ""
            else:
                return "fail", result.stderr[-500:]
        except subprocess.TimeoutExpired:
            return "fail", "Execution timed out"

    # ------------------------------------------------------------------
    # Framework startup tests
    # ------------------------------------------------------------------

    def _run_framework_startup(self, framework: str, config: Dict[str, Any]) -> List[Dict]:
        """Test if a framework can start up successfully."""
        timeout = config.get("startup_timeout", 120)

        if framework == "megatron":
            return self._test_megatron_startup(config, timeout)
        elif framework == "vllm":
            return self._test_vllm_startup(config, timeout)
        elif framework == "infinilm":
            return self._test_infinilm_startup(config, timeout)
        else:
            return [
                {
                    "name": f"compatibility.{framework}.startup",
                    "value": "unsupported",
                    "type": "scalar",
                    "unit": "",
                }
            ]

    def _test_megatron_startup(
        self, config: Dict[str, Any], timeout: int
    ) -> List[Dict]:
        """Test Megatron-LM can start a minimal training run."""
        megatron_path = config.get("megatron_path", "")
        if not megatron_path:
            return [
                {
                    "name": "compatibility.megatron.startup",
                    "value": "skipped",
                    "type": "scalar",
                    "unit": "",
                    "error": "megatron_path not configured",
                }
            ]

        script = f"{megatron_path}/pretrain_gpt.py"
        cmd = [
            "python3", script,
            "--num-layers=1", "--hidden-size=128", "--num-attention-heads=4",
            "--seq-length=128", "--max-position-embeddings=128",
            "--micro-batch-size=1", "--train-iters=2",
            "--log-interval=1", "--fp16",
            "--mock-data", "--tokenizer-type", "NullTokenizer",
            "--transformer-impl", "local",
        ]

        return self._exec_startup_test("megatron", cmd, timeout)

    def _test_vllm_startup(
        self, config: Dict[str, Any], timeout: int
    ) -> List[Dict]:
        """Test vLLM can load a model and perform one inference."""
        model = config.get("model", "facebook/opt-125m")
        cmd = [
            "python3", "-c",
            f"from vllm import LLM; llm = LLM(model='{model}'); "
            f"output = llm.generate(['Hello']); print('vLLM startup OK')",
        ]
        return self._exec_startup_test("vllm", cmd, timeout)

    def _test_infinilm_startup(
        self, config: Dict[str, Any], timeout: int
    ) -> List[Dict]:
        """Test InfiniLM can start and serve one request."""
        model = config.get("model", "facebook/opt-125m")
        cmd = [
            "python3", "-c",
            f"from infinilm import Engine; "
            f"engine = Engine(model='{model}'); "
            f"output = engine.generate('Hello'); print('InfiniLM startup OK')",
        ]
        return self._exec_startup_test("infinilm", cmd, timeout)

    def _exec_startup_test(
        self, framework: str, cmd: list, timeout: int
    ) -> List[Dict]:
        """Execute a startup test command and return metrics."""
        logger.info("Testing %s startup: %s", framework, " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            success = result.returncode == 0
            error_msg = ""
            if not success:
                error_msg = (result.stderr or result.stdout)[-500:]

            return [
                {
                    "name": f"compatibility.{framework}.startup",
                    "value": "pass" if success else "fail",
                    "type": "scalar",
                    "unit": "",
                },
                {
                    "name": f"compatibility.{framework}.startup_error",
                    "value": error_msg,
                    "type": "detail",
                    "unit": "",
                },
            ]
        except subprocess.TimeoutExpired:
            return [
                {
                    "name": f"compatibility.{framework}.startup",
                    "value": "timeout",
                    "type": "scalar",
                    "unit": "",
                },
            ]
        except FileNotFoundError as e:
            return [
                {
                    "name": f"compatibility.{framework}.startup",
                    "value": "not_found",
                    "type": "scalar",
                    "unit": "",
                    "error": str(e),
                },
            ]

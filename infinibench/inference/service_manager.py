#!/usr/bin/env python3
"""
Service Manager for Inference Frameworks
Responsible for starting, stopping, and monitoring services
"""

import subprocess
import threading
import time
import socket
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import signal
import sys
import os

logger = logging.getLogger(__name__)

# Constant definitions
DEFAULT_SERVICE_PORT = 8000
DEFAULT_MAX_WAIT_TIME = 120  # Maximum wait time (seconds)
DEFAULT_WAIT_INTERVAL = 3  # Wait interval (seconds)
DEFAULT_SERVICE_TIMEOUT = 30  # Service stop timeout (seconds)
DEFAULT_SERVICE_MAX_BATCH = 8


def _get(obj, key, default=None):
    """dict/obj safe get"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_nested(obj, path, default=None):
    """safe get for nested like 'parallel.tp'"""
    cur = obj
    for p in path.split("."):
        cur = _get(cur, p, None)
        if cur is None:
            return default
    return cur


class BaseServiceManager:
    """Base service manager abstract class"""

    def __init__(self, config: Any):
        self.config = config
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = DEFAULT_SERVICE_PORT
        self.service_started = False
        self.output_thread: Optional[threading.Thread] = None

    def start_service(self, port: int = DEFAULT_SERVICE_PORT) -> None:
        """Start service (abstract method)"""
        raise NotImplementedError

    def _get_start_timeout_s(self) -> int:
        ia = getattr(self.config, "infer_args", {}) or {}
        # Allow user to configure in infer_args
        t = _get(ia, "service_start_timeout_s", None)
        if t is None:
            # vLLM usually needs a longer startup time;
            # InfiniLM can keep a shorter default
            if str(getattr(self.config, "framework", "")).lower() == "vllm":
                return 600
            return DEFAULT_MAX_WAIT_TIME
        try:
            return int(t)
        except Exception:
            return 600

    def stop_service(self) -> None:
        """Stop service"""
        if self.server_process:
            logger.info("Stopping inference service")

            try:
                # Graceful shutdown
                self.server_process.terminate()
                self.server_process.wait(timeout=DEFAULT_SERVICE_TIMEOUT)
                logger.info("Service stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Service did not stop gracefully, forcing kill")
                self.server_process.kill()
                self.server_process.wait()

            self.server_process = None

        self.service_started = False

    def is_service_ready(self, port: Optional[int] = None) -> bool:
        """Check whether the service is ready"""
        if not self.service_started or not self.server_process:
            return False

        # Check if process is alive
        if self.server_process.poll() is not None:
            return_code = self.server_process.returncode
            logger.error(f"Server process died with return code: {return_code}")
            return False

        # Check if port is open
        check_port = port or self.server_port
        return self._check_port_open(check_port)

    def get_service_url(self) -> str:
        fw = str(getattr(self.config, "framework", "")).lower()
        if fw == "vllm":
            return f"http://localhost:{self.server_port}/v1"
        return f"http://localhost:{self.server_port}"

    def wait_for_service_ready(
        self,
        timeout: int = DEFAULT_MAX_WAIT_TIME,
        check_interval: int = DEFAULT_WAIT_INTERVAL,
    ) -> bool:
        """Wait until the service is ready"""
        logger.info("Waiting for service to be ready...")

        max_checks = timeout // check_interval
        for i in range(max_checks):
            if self.server_process and self.server_process.poll() is not None:
                rc = self.server_process.returncode
                raise RuntimeError(
                    f"Service process exited early (rc={rc}). See service logs above."
                )

            # Normal readiness check
            if self.is_service_ready():
                logger.info("Service is ready")
                return True

            elapsed_time = (i + 1) * check_interval
            logger.debug(f"Waiting... ({elapsed_time}s elapsed)")
            time.sleep(check_interval)

        logger.error(f"Service failed to start within {timeout} seconds")
        return False

    def _check_port_open(self, port: int) -> bool:
        """Check whether a port is open"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Port check failed: {e}")
            return False

    def _start_output_reader(self, process_stdout):
        """Start output reader thread"""

        def read_output():
            for line in process_stdout:
                logger.info(f"[Service] {line.strip()}")

        self.output_thread = threading.Thread(target=read_output, daemon=True)
        self.output_thread.start()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure service is stopped"""
        self.stop_service()


class InfiniLMServiceManager(BaseServiceManager):
    """InfiniLM service manager"""

    def _locate_infinilm_scripts(self) -> Optional[Path]:
        """Locate InfiniLM scripts directory"""
        search_paths = []

        # Environment variable
        env_path = os.environ.get("INFINILM_PATH")
        if env_path:
            search_paths.append(Path(env_path))

        # Current working directory
        search_paths.append(Path.cwd())

        # Common locations
        search_paths.extend(
            [
                Path.home() / "InfiniLM",
                Path("/opt/InfiniLM"),
                Path("/usr/local/InfiniLM"),
            ]
        )

        # Search
        for base_path in search_paths:
            scripts_dir = base_path / "scripts"
            if scripts_dir.exists():
                logger.info(f"Found InfiniLM scripts at: {scripts_dir}")
                return scripts_dir

        logger.error("Cannot locate InfiniLM scripts directory")
        return None

    def _build_start_command(self, launch_script: Path, port: int) -> List[str]:
        """Build start command (dict/obj compatible)"""

        infer_args = getattr(self.config, "infer_args", {})
        device = getattr(self.config, "device", {})
        tp = _get_nested(infer_args, "parallel.tp", 1)
        max_seq_len = _get(infer_args, "max_seq_len", 4096)

        max_batch = _get(infer_args, "max_batch", None)
        if max_batch is None:
            max_batch = _get(infer_args, "service_max_batch", 4)
        try:
            max_batch = int(max_batch)
        except Exception:
            max_batch = 4

        accelerator = _get_nested(device, "accelerator.value", None)
        if accelerator is None:
            accelerator = _get(device, "accelerator", "nvidia")
        accelerator = str(accelerator).lower()

        cmd = [
            sys.executable,
            str(launch_script),
            "--model-path",
            str(self.config.model_path),
            "--ndev",
            str(tp),
            "--max-batch",
            str(max_batch),
            "--dev",
            accelerator,
        ]

        if port != DEFAULT_SERVICE_PORT:
            cmd.extend(["--port", str(port)])

        if max_seq_len and int(max_seq_len) != 4096:
            cmd.extend(["--max-tokens", str(max_seq_len)])

        return cmd

    def start_service(self, port: int = DEFAULT_SERVICE_PORT) -> None:
        """Start InfiniLM service"""
        logger.info(f"Starting InfiniLM service on port {port}")

        # Locate scripts directory
        scripts_dir = self._locate_infinilm_scripts()
        if not scripts_dir:
            raise RuntimeError("Cannot locate InfiniLM scripts directory")

        launch_script = scripts_dir / "launch_server.py"
        if not launch_script.exists():
            raise FileNotFoundError(f"Launch script not found: {launch_script}")

        # Build start command
        cmd = self._build_start_command(launch_script, port)

        # Start service process
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        self.server_port = port
        self.service_started = True

        # Start output reader thread
        self._start_output_reader(self.server_process.stdout)

        logger.info(f"InfiniLM service started with PID: {self.server_process.pid}")
        logger.info(f"Command: {' '.join(cmd)}")

        # 5. Wait for service to be ready
        if not self.wait_for_service_ready():
            self.stop_service()
            raise TimeoutError("InfiniLM service failed to start within timeout")


class VLLMServiceManager(BaseServiceManager):
    """vLLM service manager"""

    def _build_start_command(self, port: int) -> List[str]:
        """Build vLLM API server start command (dict/obj compatible)"""

        infer_args = getattr(self.config, "infer_args", {})
        device = getattr(self.config, "device", {})

        tp = _get_nested(infer_args, "parallel.tp", 1)
        max_seq_len = _get(infer_args, "max_seq_len", None)

        accelerator = _get_nested(device, "accelerator.value", None)
        if accelerator is None:
            accelerator = _get(device, "accelerator", "nvidia")
        accelerator = str(accelerator).lower()

        cpu_only = bool(_get(device, "cpu_only", False))

        # framework_kwargs support
        framework_kwargs = _get(infer_args, "framework_kwargs", None)
        if not isinstance(framework_kwargs, dict):
            framework_kwargs = {}

        # dtype: use user-provided value if specified,
        # otherwise default to "auto"
        dtype = framework_kwargs.get("dtype", "auto")

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.config.model_path),
            "--served-model-name",
            str(self.config.model),
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp),
            "--dtype",
            str(dtype),
            "--disable-log-stats",
        ]

        if max_seq_len:
            cmd.extend(["--max-model-len", str(max_seq_len)])

        if (not cpu_only) and accelerator == "nvidia":
            from infinibench.common.constants import (
                DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
                DEFAULT_VLLM_SWAP_SPACE,
            )

            gmu = framework_kwargs.get(
                "gpu_memory_utilization", DEFAULT_VLLM_GPU_MEMORY_UTILIZATION
            )
            swp = framework_kwargs.get("swap_space", DEFAULT_VLLM_SWAP_SPACE)

            cmd.extend(
                [
                    "--gpu-memory-utilization",
                    str(gmu),
                    "--swap-space",
                    str(swp),
                ]
            )
        else:
            logger.warning(
                f"vLLM accelerator='{accelerator}', cpu_only={cpu_only}; may not be fully supported."
            )

        if framework_kwargs:
            param_mapping = {
                "trust_remote_code": "trust-remote-code",
                "max_num_batched_tokens": "max-num-batched-tokens",
                "max_num_seqs": "max-num-seqs",
                "enforce_eager": "enforce-eager",
                "quantization": "quantization",
                "block_size": "block-size",
                "max_logprobs": "max-logprobs",
                "seed": "seed",
            }

            BOOL_FLAGS = {"trust-remote-code", "enforce-eager"}

            for config_param, cmd_param in param_mapping.items():
                if config_param not in framework_kwargs:
                    continue
                val = framework_kwargs.get(config_param, None)
                if val is None:
                    continue

                flag = f"--{cmd_param}"

                if cmd_param in BOOL_FLAGS:
                    if isinstance(val, bool):
                        if val:
                            cmd.append(flag)
                    else:
                        sval = str(val).strip().lower()
                        if sval in ("1", "true", "yes", "y", "on"):
                            cmd.append(flag)
                    continue

                cmd.extend([flag, str(val)])

        return cmd

    def start_service(self, port: int = DEFAULT_SERVICE_PORT) -> None:
        """Start vLLM API server"""
        logger.info(f"Starting vLLM API server on port {port}")

        # Build the startup command
        cmd = self._build_start_command(port)

        logger.info(f"vLLM API server command: {' '.join(cmd)}")

        # Start the service process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        self.server_port = port
        self.service_started = True

        # Start the output reader thread
        self._start_output_reader(self.server_process.stdout)

        logger.info(f"vLLM API server started with PID: {self.server_process.pid}")

        # Wait for the service to become ready
        timeout_s = self._get_start_timeout_s()
        if not self.wait_for_service_ready(timeout=timeout_s):
            self.stop_service()
            raise TimeoutError("vLLM API server failed to start within timeout")

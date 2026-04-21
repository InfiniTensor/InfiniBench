#!/usr/bin/env python3
"""NCCL Communication Test Adapter"""

import csv
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from infinibench.adapter import BaseAdapter

logger = logging.getLogger(__name__)

# -----------------------------
# Constants / Defaults
# -----------------------------
DEFAULTS = {
    "min_size": "8",
    "max_size": "128M",
    "step_factor": "2",
    "timeout_ms": 300000,
    "warmup_iterations": 10,
    "measured_iterations": 100,
    "device_involved": 8,
    "output_dir": "./output/comm",
    "nccl_env_defaults": {
        "NCCL_DEBUG": "INFO",
        "NCCL_IB_DISABLE": "0",
    },
}


@dataclass(frozen=True)
class NCCLTestsSpec:
    operation: str
    binary_name: str
    display_name: str


OPERATION_MAP: Dict[str, NCCLTestsSpec] = {
    "allreduce": NCCLTestsSpec("allreduce", "all_reduce_perf", "AllReduce"),
    "allgather": NCCLTestsSpec("allgather", "all_gather_perf", "AllGather"),
    "alltoall": NCCLTestsSpec("alltoall", "alltoall_perf", "AllToAll"),
    "broadcast": NCCLTestsSpec("broadcast", "broadcast_perf", "Broadcast"),
    "pointtopoint": NCCLTestsSpec("pointtopoint", "sendrecv_perf", "PointToPoint"),
}


@dataclass
class ResolvedRun:
    mode: str = "single_node"  # "single_node" | "multi_node"
    device_used: int = 0  # Actual number of GPUs used (single node: -g n)
    gpus_per_node: int = 0  # Number of GPUs used per node in multi-node mode
    nodes: int = 1  # Number of nodes in multi-node mode
    command: str = ""


class NcclTestsAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.nccl_test_dir: Optional[Path] = None
        self.result_dir: Optional[Path] = None
        self.test_spec: Optional[NCCLTestsSpec] = None
        self.run_id: Optional[str] = None

        self.resolved = ResolvedRun()
        self._orig_env: Dict[str, str] = {}

    # -----------------------------
    # BaseAdapter hooks
    # -----------------------------
    def setup(self, config: Dict[str, Any]) -> None:
        self._save_orig_env(["CUDA_VISIBLE_DEVICES"])

        self.nccl_test_dir = self._find_nccl_test_dir()
        if not self.nccl_test_dir:
            raise FileNotFoundError(
                "NCCL tests directory not found (expected submodules/nccl-tests)."
            )

        out_dir = config.get("output_dir", DEFAULTS["output_dir"])
        self.result_dir = Path(out_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self._setup_nccl_env(config)
        logger.info(f"NCCL tests found at: {self.nccl_test_dir}")

    def teardown(self) -> None:
        self._restore_orig_env()
        logger.info("NCCL adapter teardown complete")

    def process(self, test_input: Dict[str, Any]) -> Dict[str, Any]:
        input_dict = (
            test_input.to_dict() if hasattr(test_input, "to_dict") else test_input
        )
        config = input_dict.get("config", {}) or {}
        testcase = input_dict.get("testcase", "") or ""

        self.run_id = input_dict.get("run_id") or self._gen_run_id(testcase)
        self.test_spec = self._parse_test_spec(testcase)
        if not self.test_spec:
            raise ValueError(f"Unknown operation in testcase: {testcase}")

        try:
            cmd = self._build_command(config)
            cmd_str = " ".join(cmd)
            self.resolved.command = cmd_str

            t0 = time.perf_counter()
            stdout, stderr, rc = self._run(cmd, config)
            wall_ms = (time.perf_counter() - t0) * 1000.0

            results = self._parse_output(stdout)
            if not results["latency"]:
                msg = f"No performance data parsed. returncode={rc}"
                if stderr:
                    msg += "\nStderr(last 20 lines):\n" + "\n".join(
                        stderr.splitlines()[-20:]
                    )
                raise RuntimeError(msg)

            raw_files = self._save_raw_csv(results)
            metrics = self._build_metrics(wall_ms, raw_files)

            return {
                "run_id": self.run_id,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "testcase": testcase,
                "success": 0,
                "config": self._build_config_section(config, cmd_str),
                "metrics": metrics,
                "result_code": 0,
                "resolved": {
                    "mode": self.resolved.mode,
                    "nodes": self.resolved.nodes,
                    "gpus_per_node": self.resolved.gpus_per_node,
                    "device_used": self.resolved.device_used,
                    "command": self.resolved.command,
                },
            }

        except Exception as e:
            # Log error with context, then re-raise for Executor to handle
            operation = (
                self.test_spec.get("op", "unknown") if self.test_spec else "unknown"
            )
            logger.error(
                f"NCCLAdapter: Test failed for {testcase}\n"
                f"  Operation: {operation}\n"
                f"  Nodes: {self.resolved.nodes}\n"
                f"  GPUs per node: {self.resolved.gpus_per_node}\n"
                f"  Error: {str(e)}",
                exc_info=True,
            )
            raise

    # -----------------------------
    # Config helpers
    # -----------------------------
    def _cfg(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """
        Unified config getter:
        - if config.single_node exists, allow override there for single-node mode
        - if config.multi_node exists, allow override there for multi-node mode
        """
        mode = "multi_node" if self._is_multi_node(config) else "single_node"
        sub = config.get(mode) or {}
        return sub.get(key, config.get(key, default))

    def _is_multi_node(self, config: Dict[str, Any]) -> bool:
        mn = config.get("multi_node")
        return isinstance(mn, dict) and bool(mn)

    # -----------------------------
    # Environment / HW collection
    # -----------------------------
    def _save_orig_env(self, keys: List[str]) -> None:
        for k in keys:
            if k in os.environ:
                self._orig_env[k] = os.environ[k]

    def _restore_orig_env(self) -> None:
        # restore existing
        for k, v in self._orig_env.items():
            os.environ[k] = v
        # remove those we added
        for k in ["CUDA_VISIBLE_DEVICES"]:
            if k not in self._orig_env and k in os.environ:
                del os.environ[k]

    def _setup_nccl_env(self, config: Dict[str, Any]) -> None:
        # CUDA_VISIBLE_DEVICES
        device_ids = config.get("device_ids")
        if device_ids is None and isinstance(config.get("single_node"), dict):
            device_ids = config["single_node"].get("device_ids")

        if isinstance(device_ids, list) and device_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in device_ids)
            logger.info(
                f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
            )

        env_vars: Dict[str, Any] = {}
        env_vars.update(DEFAULTS["nccl_env_defaults"])
        env_vars.update(config.get("env_vars", {}) or {})
        for k, v in env_vars.items():
            os.environ[k] = str(v)

    # -----------------------------
    # NCCL tests dir / binary
    # -----------------------------
    def _find_nccl_test_dir(self) -> Optional[Path]:
        candidates = [
            Path.cwd() / "submodules" / "nccl-tests",
            Path(__file__).resolve().parents[2] / "submodules" / "nccl-tests",
        ]
        for p in candidates:
            if (p / "build" / "all_reduce_perf").exists():
                return p
        return None

    def _find_binary(self, name: str) -> Path:
        assert self.nccl_test_dir is not None
        p = self.nccl_test_dir / "build" / name
        if p.exists():
            return p
        p2 = self.nccl_test_dir / name
        if p2.exists():
            return p2
        which = shutil.which(name)
        if which:
            return Path(which)
        raise FileNotFoundError(f"NCCL binary '{name}' not found")

    # -----------------------------
    # testcase / command
    # -----------------------------
    def _parse_test_spec(self, testcase: str) -> Optional[NCCLTestsSpec]:
        # comm.NcclTest.AllReduce
        parts = testcase.split(".")
        if len(parts) < 3:
            return None
        op = parts[2].lower()
        if "point" in op:
            op = "pointtopoint"
        return OPERATION_MAP.get(op)

    def _build_command(self, config: Dict[str, Any]) -> List[str]:
        assert self.test_spec is not None
        binary = self._find_binary(self.test_spec.binary_name)

        min_size = self._cfg(config, "min_size", DEFAULTS["min_size"])
        max_size = self._cfg(config, "max_size", DEFAULTS["max_size"])
        step = self._cfg(config, "step_factor", DEFAULTS["step_factor"])
        warm = int(
            self._cfg(config, "warmup_iterations", DEFAULTS["warmup_iterations"])
        )
        meas = int(
            self._cfg(config, "measured_iterations", DEFAULTS["measured_iterations"])
        )

        if self._is_multi_node(config):
            mn = config.get("multi_node") or {}
            hosts = mn.get("hosts") or []
            gpn = int(mn.get("gpus_per_node", DEFAULTS["device_involved"]))
            extra_mpi = mn.get("extra_mpi_args") or []
            mpirun = mn.get("mpirun", "mpirun")

            if not hosts:
                raise ValueError("multi_node.hosts is required for multi-node run")

            host_arg = ",".join(f"{h}:{gpn}" for h in hosts)

            # record resolved
            self.resolved.mode = "multi_node"
            self.resolved.nodes = len(hosts)
            self.resolved.gpus_per_node = gpn
            self.resolved.device_used = len(hosts) * gpn
            # NCCL tests: USE-G to control the number of gpus per node when there is one process per node
            return [
                mpirun,
                "-H",
                host_arg,
                "-np",
                str(len(hosts)),
                *[str(x) for x in extra_mpi],
                str(binary),
                "-b",
                str(min_size),
                "-e",
                str(max_size),
                "-f",
                str(step),
                "-g",
                str(gpn),
                *self._op_extra_args(),
                "-w",
                str(warm),
                "-n",
                str(meas),
                *self._extra_nccl_args(config),
            ]

        # single node
        device_involved = int(
            self._cfg(config, "device_involved", DEFAULTS["device_involved"])
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        visible_cnt = (
            len([x for x in visible.split(",") if x.strip()]) if visible else 0
        )
        g_used = min(device_involved, visible_cnt or device_involved)

        self.resolved.mode = "single_node"
        self.resolved.nodes = 1
        self.resolved.gpus_per_node = g_used
        self.resolved.device_used = g_used

        return [
            str(binary),
            "-b",
            str(min_size),
            "-e",
            str(max_size),
            "-f",
            str(step),
            "-g",
            str(g_used),
            *self._op_extra_args(),
            "-w",
            str(warm),
            "-n",
            str(meas),
            *self._extra_nccl_args(config),
        ]

    def _op_extra_args(self) -> List[str]:
        assert self.test_spec is not None
        if self.test_spec.operation == "allreduce":
            return ["-c", "1"]
        return []

    def _extra_nccl_args(self, config: Dict[str, Any]) -> List[str]:
        args = config.get("nccl_args") or []
        return [str(x) for x in args]

    # -----------------------------
    # run / parse
    # -----------------------------
    def _run(self, cmd: List[str], config: Dict[str, Any]) -> Tuple[str, str, int]:
        timeout_ms = int(self._cfg(config, "timeout_ms", DEFAULTS["timeout_ms"]))
        timeout_s = max(1, timeout_ms // 1000)

        assert self.nccl_test_dir is not None
        run_dir = self.nccl_test_dir / "build"
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(run_dir),
            timeout=timeout_s,
            env=os.environ,
        )
        return r.stdout or "", r.stderr or "", r.returncode

    def _parse_output(self, stdout: str) -> Dict[str, Any]:
        out = {"latency": [], "busbw": [], "algbw": [], "summary": {}}
        if not stdout:
            return out

        lat, bus = [], []

        for line in stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "NCCL INFO" in line:
                continue
            if not line[0].isdigit():
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            # nccl-tests typical format:
            # size count type redop root time algbw busbw ...
            try:
                size_b = self._parse_size(parts[0])
                t_us = float(parts[5])
                algbw = float(parts[6])
                busbw = float(parts[7])
            except Exception:
                continue

            out["latency"].append({"size_bytes": size_b, "latency_us": t_us})
            out["algbw"].append({"size_bytes": size_b, "algbw_gbs": algbw})
            out["busbw"].append({"size_bytes": size_b, "busbw_gbs": busbw})
            lat.append(t_us)
            bus.append(busbw)

        if lat:
            out["summary"]["avg_latency_us"] = sum(lat) / len(lat)
            out["summary"]["max_latency_us"] = max(lat)
            out["summary"]["num_samples"] = len(lat)
        if bus:
            out["summary"]["avg_busbw_gbs"] = sum(bus) / len(bus)
            out["summary"]["max_busbw_gbs"] = max(bus)

        return out

    def _parse_size(self, s: str) -> int:
        s = s.strip().upper()
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        if s and s[-1] in mult:
            return int(float(s[:-1]) * mult[s[-1]])
        return int(s)

    # -----------------------------
    # output files / schema
    # -----------------------------
    def _save_raw_csv(self, results: Dict[str, Any]) -> Dict[str, str]:
        assert self.result_dir is not None
        assert self.run_id is not None

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = re.sub(r"\d{8}_\d{6}", "", self.run_id).strip("._") or "comm_test"
        prefix = f"{base}_{ts}"

        raw: Dict[str, str] = {}

        if results["latency"]:
            fp = self.result_dir / f"{prefix}_latency.csv"
            with fp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["size_bytes", "latency_us"])
                for x in results["latency"]:
                    w.writerow([x["size_bytes"], x["latency_us"]])
            raw["latency"] = f"./{fp.name}"

        if results["busbw"]:
            fp = self.result_dir / f"{prefix}_bandwidth.csv"
            with fp.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["size_bytes", "bandwidth_gbs"])
                for x in results["busbw"]:
                    w.writerow([x["size_bytes"], x["busbw_gbs"]])
            raw["bandwidth"] = f"./{fp.name}"

        return raw

    def _build_metrics(
        self, duration_ms: float, raw_files: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        if raw_files.get("latency"):
            metrics.append(
                {
                    "name": "comm.latency",
                    "type": "timeseries",
                    "raw_data_url": raw_files["latency"],
                    "unit": "us",
                }
            )
        metrics.append(
            {
                "name": "comm.duration",
                "type": "scalar",
                "value": duration_ms,
                "unit": "ms",
            }
        )
        if raw_files.get("bandwidth"):
            metrics.append(
                {
                    "name": "comm.bandwidth",
                    "type": "timeseries",
                    "raw_data_url": raw_files["bandwidth"],
                    "unit": "GB/s",
                }
            )
        return metrics

    def _build_config_section(
        self, config: Dict[str, Any], command: str
    ) -> Dict[str, Any]:
        assert self.test_spec is not None
        core = {
            "command": command,
            "operator": config.get("operator", self.test_spec.display_name),
            "attributes": config.get(
                "attributes",
                [{"name": "op", "value": "SUM"}, {"name": "group", "value": "WORLD"}],
            ),
            "inputs": config.get(
                "inputs",
                [{"name": "input_tensor", "dtype": "float32", "shape": [512, 512]}],
            ),
            "outputs": config.get(
                "outputs",
                [{"name": "output_tensor", "dtype": "float32", "shape": [512, 512]}],
            ),
            "timeout_ms": self._cfg(config, "timeout_ms", DEFAULTS["timeout_ms"]),
            "device_involved": int(
                self._cfg(config, "device_involved", DEFAULTS["device_involved"])
            ),
            "device_used": int(self.resolved.device_used or 0),
            "warmup_iterations": int(
                self._cfg(config, "warmup_iterations", DEFAULTS["warmup_iterations"])
            ),
            "measured_iterations": int(
                self._cfg(
                    config, "measured_iterations", DEFAULTS["measured_iterations"]
                )
            ),
        }

        extras = dict(config) if isinstance(config, dict) else {}
        for k in list(core.keys()):
            extras.pop(k, None)

        # Optional
        if extras.get("multi_node", None) is None:
            extras.pop("multi_node", None)

        # Merger
        out = {}
        out.update(core)
        out.update(extras)
        return out

    # -----------------------------
    # utils
    # -----------------------------
    def _err(self, input_dict: Dict[str, Any], msg: str) -> Dict[str, Any]:
        return {
            "run_id": input_dict.get("run_id", f"comm.error.{uuid.uuid4().hex[:8]}"),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "testcase": input_dict.get("testcase", ""),
            "success": 1,
            "result_code": 1,
            "config": input_dict.get("config", {}),
            "metrics": [],
            "error": msg,
        }

    def _gen_run_id(self, testcase: str) -> str:
        op = testcase.split(".")[-1] if testcase else "unknown"
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"comm.{op}.{ts}.{uuid.uuid4().hex[:8]}"

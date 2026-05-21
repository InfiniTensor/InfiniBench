#!/usr/bin/env python3
"""Stability Test Adapter.

Runs Megatron-LM training for a configurable number of steps, saves a
checkpoint at a specified interval, restarts from that checkpoint, and
analyzes loss continuity.

Testcase format:
    stability.Megatron.LongRun

Config fields:
    train_iters: Total training iterations (default 2000)
    save_interval: Checkpoint save interval (default 1000)
    restart_from: Iteration to restart from (default 1000)
    spike_threshold: Z-score threshold for spike detection (default 5.0)
    loss_window: Rolling window for spike detection (default 10)
    + all standard TrainingAdapter/Megatron config fields
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.constants import InfiniMetricsJson, ErrorCode
from infinimetrics.stability.loss_analyzer import analyze_loss_series
from infinimetrics.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)


class StabilityAdapter(BaseAdapter):
    """Adapter for long-run stability testing with checkpoint restart."""

    # Regex patterns for parsing Megatron output
    _ITER_PATTERN = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    _LOSS_PATTERN = re.compile(
        r"lm loss:\s*([+\-]?\d+(?:\.\d+)?(?:[Ee][+\-]?\d+)?)", re.IGNORECASE
    )
    _THROUGHPUT_PATTERN = re.compile(
        r"elapsed time per iteration \(ms\):\s*([0-9]*\.?[0-9]+)", re.IGNORECASE
    )
    _SAVE_PATTERN = re.compile(
        r"successfully saved checkpoint at iteration\s+(\d+)", re.IGNORECASE
    )

    def __init__(self):
        self.config = {}

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources."""
        self.config = config

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Execute two-phase stability test."""
        test_dict = self._normalize_test_input(test_input)
        if not test_dict:
            return self._create_error_response(
                "Invalid test input format", result_code=ErrorCode.CONFIG
            )

        testcase = test_dict.get(InfiniMetricsJson.TESTCASE, "unknown")
        config = test_dict.get(InfiniMetricsJson.CONFIG, {})
        run_id = test_dict.get(InfiniMetricsJson.RUN_ID, "unknown")

        logger.info(f"StabilityAdapter: Processing {testcase}")

        train_iters = config.get("train_iters", 2000)
        save_interval = config.get("save_interval", 1000)
        restart_from = config.get("restart_from", 1000)

        output_dir = Path(config.get("output_dir", "./output")) / "stability"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Phase A: Full training run
            logger.info(
                "Phase A: Training for %d iterations with save_interval=%d",
                train_iters,
                save_interval,
            )
            phase_a = self._run_training(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                phase="phase_a",
                train_iters=train_iters,
                save_interval=save_interval,
                extra_args=[],
            )

            # Phase B: Restart from checkpoint
            ckpt_dir = phase_a.get("checkpoint_dir")
            if not ckpt_dir:
                raise RuntimeError(
                    "Phase A completed but no checkpoint was saved. "
                    "Cannot proceed to Phase B restart."
                )

            logger.info(
                "Phase B: Restarting from checkpoint at %s", ckpt_dir
            )
            phase_b = self._run_training(
                config=config,
                output_dir=output_dir,
                run_id=run_id,
                phase="phase_b",
                train_iters=train_iters,
                save_interval=train_iters + 1,  # Don't save again
                extra_args=[f"--load={ckpt_dir}"],
            )

            # Analyze combined results
            metrics = self._build_stability_metrics(
                phase_a, phase_b, config, run_id
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
                f"StabilityAdapter: Test failed for {testcase}: {e}",
                exc_info=True,
            )
            raise

    def _run_training(
        self,
        config: Dict[str, Any],
        output_dir: Path,
        run_id: str,
        phase: str,
        train_iters: int,
        save_interval: int,
        extra_args: List[str],
    ) -> Dict[str, Any]:
        """Execute one phase of the stability test.

        Returns dict with losses_by_iter, checkpoint_dir, last_iter, log_file.
        """
        cmd = self._build_megatron_command(
            config, train_iters, save_interval, extra_args
        )

        log_file = output_dir / f"{run_id}_{phase}.log"
        losses_csv = output_dir / f"{run_id}_{phase}_loss.csv"
        throughput_csv = output_dir / f"{run_id}_{phase}_throughput.csv"

        logger.info("Launching %s: %s", phase, " ".join(cmd))

        metrics = {
            "losses_by_iter": {},
            "throughput_by_iter": {},
            "last_seen_iter": None,
            "checkpoint_dir": None,
            "saved_iterations": [],
        }

        env = self._get_env(config)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        with open(log_file, "w") as f:
            for line in proc.stdout:
                line = line.rstrip("\n")
                logger.debug("[%s] %s", phase, line)
                f.write(line + "\n")
                self._parse_output_line(line, metrics, config)

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(
                f"{phase} training failed (code {proc.returncode}). "
                f"Log: {log_file}"
            )

        # Find checkpoint directory
        ckpt_iter = None
        for it in metrics["saved_iterations"]:
            if it <= train_iters:
                ckpt_iter = it

        if ckpt_iter is not None:
            save_dir = config.get("save_dir", str(output_dir / "checkpoints"))
            metrics["checkpoint_dir"] = save_dir

        # Save CSVs
        self._save_csv(losses_csv, metrics["losses_by_iter"], "iteration,loss")
        self._save_csv(
            throughput_csv, metrics["throughput_by_iter"], "iteration,throughput"
        )

        metrics["log_file"] = str(log_file)
        metrics["loss_csv"] = str(losses_csv)
        metrics["throughput_csv"] = str(throughput_csv)

        return metrics

    def _build_megatron_command(
        self,
        config: Dict[str, Any],
        train_iters: int,
        save_interval: int,
        extra_args: List[str],
    ) -> List[str]:
        """Build the Megatron torchrun command."""
        import random

        megatron_path = config.get("megatron_path", "")
        train_script = (
            f"{megatron_path}/pretrain_gpt.py" if megatron_path else "pretrain_gpt.py"
        )

        train_args = config.get("train_args", {})
        parallel = train_args.get("parallel", {})
        tp = parallel.get("tp", 1)
        pp = parallel.get("pp", 1)
        dp = parallel.get("dp", 1)

        # Try to get device count
        device_count = 1
        device_config = config.get("device", {})
        if device_config.get("device_ids"):
            device_count = len(device_config["device_ids"])

        nproc = min(dp * tp * max(1, pp), device_count) if device_count > 0 else dp * tp * pp

        model_config = train_args.get("model", {})
        num_layers = model_config.get("num_layers", train_args.get("num_layers", 12))
        hidden_size = model_config.get("hidden_size", train_args.get("hidden_size", 768))
        num_heads = model_config.get(
            "num_attention_heads", train_args.get("num_attention_heads", 12)
        )
        seq_len = train_args.get("seq_len", 1024)
        mbs = train_args.get("mbs", 1)

        save_dir = config.get("save_dir", "./checkpoints")

        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--master_port={random.randint(20000, 60000)}",
            train_script,
            f"--tensor-model-parallel-size={tp}",
            f"--pipeline-model-parallel-size={pp}",
            f"--num-layers={num_layers}",
            f"--hidden-size={hidden_size}",
            f"--num-attention-heads={num_heads}",
            f"--seq-length={seq_len}",
            f"--max-position-embeddings={seq_len}",
            f"--micro-batch-size={mbs}",
            f"--train-iters={train_iters}",
            f"--save-interval={save_interval}",
            f"--save={save_dir}",
            "--transformer-impl", "local",
            "--no-gradient-accumulation-fusion",
            "--log-interval", "1",
            "--log-throughput",
            "--mock-data", "--tokenizer-type", "NullTokenizer",
        ]

        precision = train_args.get("precision", "fp16")
        if precision == "bf16":
            cmd.append("--bf16")
        else:
            cmd.append("--fp16")

        lr = train_args.get("lr")
        if lr:
            cmd.append(f"--lr={lr}")

        cmd.extend(extra_args)

        return cmd

    def _get_env(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Get environment variables."""
        env = os.environ.copy()
        device_config = config.get("device", {})
        if device_ids := device_config.get("device_ids"):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
        return env

    def _parse_output_line(self, line: str, metrics: Dict, config: Dict) -> None:
        """Parse a Megatron output line for loss/throughput/checkpoint info."""
        # Iteration
        m_iter = self._ITER_PATTERN.search(line)
        if m_iter:
            try:
                metrics["last_seen_iter"] = int(m_iter.group(1))
            except ValueError:
                pass

        # Loss
        m_loss = self._LOSS_PATTERN.search(line)
        if m_loss and metrics["last_seen_iter"] is not None:
            try:
                metrics["losses_by_iter"][metrics["last_seen_iter"]] = float(
                    m_loss.group(1)
                )
            except ValueError:
                pass

        # Throughput
        m_tp = self._THROUGHPUT_PATTERN.search(line)
        if m_tp and metrics["last_seen_iter"] is not None:
            try:
                train_args = config.get("train_args", {})
                mbs = train_args.get("mbs", 1)
                seq_len = train_args.get("seq_len", 1024)
                elapsed_ms = float(m_tp.group(1))
                if elapsed_ms > 0:
                    metrics["throughput_by_iter"][
                        metrics["last_seen_iter"]
                    ] = mbs * seq_len / (elapsed_ms / 1000.0)
            except ValueError:
                pass

        # Checkpoint save
        m_save = self._SAVE_PATTERN.search(line)
        if m_save:
            try:
                metrics["saved_iterations"].append(int(m_save.group(1)))
            except ValueError:
                pass

    def _save_csv(self, path: Path, data: Dict, header: str) -> None:
        """Save a simple two-column CSV."""
        with open(path, "w") as f:
            f.write(header + "\n")
            for key, val in sorted(data.items()):
                f.write(f"{key},{val}\n")

    def _build_stability_metrics(
        self,
        phase_a: Dict,
        phase_b: Dict,
        config: Dict,
        run_id: str,
    ) -> List[Dict]:
        """Build the final metrics from both phases."""
        train_iters = config.get("train_iters", 2000)
        restart_from = config.get("restart_from", 1000)
        spike_threshold = config.get("spike_threshold", 5.0)
        loss_window = config.get("loss_window", 10)

        # Analyze Phase A loss (full training)
        phase_a_analysis = analyze_loss_series(
            phase_a["losses_by_iter"],
            spike_threshold=spike_threshold,
            spike_window=loss_window,
        )

        # Analyze Phase B loss (after restart)
        phase_b_analysis = analyze_loss_series(
            phase_b["losses_by_iter"],
            spike_threshold=spike_threshold,
            spike_window=loss_window,
        )

        # Check if checkpoint was saved
        checkpoint_saved = phase_a["checkpoint_dir"] is not None

        # Check if restart was successful (Phase B ran some iterations)
        restart_successful = len(phase_b["losses_by_iter"]) > 0

        # Check loss continuity: last loss of Phase A vs first loss of Phase B
        phase_a_losses = phase_a["losses_by_iter"]
        phase_b_losses = phase_b["losses_by_iter"]

        loss_continuous = False
        if phase_a_losses and phase_b_losses:
            last_a_loss = phase_a_losses[max(phase_a_losses.keys())]
            first_b_loss = phase_b_losses[min(phase_b_losses.keys())]
            # Continuity check: loss difference should be small
            loss_diff = abs(last_a_loss - first_b_loss)
            loss_continuous = loss_diff < 1.0  # threshold

        # Trend after restart
        post_restart_trend = phase_b_analysis["trend"]

        # Summary metrics
        metrics = [
            {
                "name": "stability.phase_a.completed_iterations",
                "value": phase_a["last_seen_iter"] or 0,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.phase_b.completed_iterations",
                "value": phase_b["last_seen_iter"] or 0,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.checkpoint_saved",
                "value": "pass" if checkpoint_saved else "fail",
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.restart_successful",
                "value": "pass" if restart_successful else "fail",
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.loss_has_nan",
                "value": phase_a_analysis["summary"]["has_nan"] or phase_b_analysis["summary"]["has_nan"],
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.loss_has_inf",
                "value": phase_a_analysis["summary"]["has_inf"] or phase_b_analysis["summary"]["has_inf"],
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.loss_spike_count",
                "value": len(phase_a_analysis["spikes"]) + len(phase_b_analysis["spikes"]),
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.loss_continuous",
                "value": "pass" if loss_continuous else "fail",
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.post_restart_loss_decreasing",
                "value": "pass" if post_restart_trend["decreasing"] else "fail",
                "type": "scalar",
                "unit": "",
            },
            {
                "name": "stability.phase_a_loss_csv",
                "value": phase_a.get("loss_csv", ""),
                "type": "file",
                "unit": "",
            },
            {
                "name": "stability.phase_b_loss_csv",
                "value": phase_b.get("loss_csv", ""),
                "type": "file",
                "unit": "",
            },
            {
                "name": "stability.phase_a_throughput_csv",
                "value": phase_a.get("throughput_csv", ""),
                "type": "file",
                "unit": "",
            },
            {
                "name": "stability.phase_b_throughput_csv",
                "value": phase_b.get("throughput_csv", ""),
                "type": "file",
                "unit": "",
            },
            {
                "name": "stability.analysis.phase_a",
                "value": phase_a_analysis,
                "type": "detail",
                "unit": "",
            },
            {
                "name": "stability.analysis.phase_b",
                "value": phase_b_analysis,
                "type": "detail",
                "unit": "",
            },
        ]

        return metrics

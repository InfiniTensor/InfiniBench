#!/usr/bin/env python3
"""Traditional Model Adapter for benchmarking classic CV/NLP/Speech/etc models.

Interfaces with the PyTorchModels/ directory to run training and evaluation
scripts for 50+ traditional models across 11 domains.

Testcase format:
    traditional.Train.<Category>/<ModelName>
    traditional.Eval.<Category>/<ModelName>
    traditional.TrainAll

Config fields:
    model_category: Category name (e.g., "Detection")
    model_name: Model name (e.g., "fasterrcnn") or "all"
    mode: "train" or "eval"
    platform: Target platform (for env.sh)
    timeout: Per-model timeout in seconds (default 600)
    pytorch_models_dir: Override PyTorchModels path
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from infinimetrics.adapter import BaseAdapter
from infinimetrics.common.constants import InfiniMetricsJson, ErrorCode
from infinimetrics.traditional_models.model_registry import (
    get_model,
    get_models_by_category,
    get_all_models,
    get_platform_env_name,
    _PYTORCH_MODELS_DIR,
)
from infinimetrics.traditional_models.log_parser import parse_log
from infinimetrics.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)


class TraditionalModelAdapter(BaseAdapter):
    """Adapter for traditional model training/evaluation benchmarks."""

    def __init__(self):
        self.config = {}

    def setup(self, config: Dict[str, Any]) -> None:
        """Initialize resources."""
        self.config = config

    def process(self, test_input: Any) -> Dict[str, Any]:
        """Execute traditional model test(s)."""
        test_dict = self._normalize_test_input(test_input)
        if not test_dict:
            return self._create_error_response(
                "Invalid test input format", result_code=ErrorCode.CONFIG
            )

        testcase = test_dict.get(InfiniMetricsJson.TESTCASE, "unknown")
        config = test_dict.get(InfiniMetricsJson.CONFIG, {})
        run_id = test_dict.get(InfiniMetricsJson.RUN_ID, "unknown")

        logger.info(f"TraditionalModelAdapter: Processing {testcase}")

        # Parse testcase for mode
        parts = testcase.split(".")
        if len(parts) < 3:
            return self._create_error_response(
                f"Invalid testcase format: {testcase}",
                test_dict,
                result_code=ErrorCode.CONFIG,
            )

        mode_raw = parts[1].lower()
        model_spec = parts[2]  # e.g., "Detection/fasterrcnn" or "All"

        # Determine mode
        if mode_raw == "train":
            mode = "train"
        elif mode_raw in ("eval", "evaluate", "inference"):
            mode = "eval"
        else:
            mode = "train"

        # Determine models to run
        models_to_run = self._resolve_models(model_spec, config)
        if not models_to_run:
            return self._create_error_response(
                f"No models found for spec: {model_spec}",
                test_dict,
                result_code=ErrorCode.CONFIG,
            )

        timeout = config.get("timeout", 600)
        platform = config.get("platform", "nvidia")
        models_dir = Path(
            config.get("pytorch_models_dir", str(_PYTORCH_MODELS_DIR))
        )

        # Execute models
        all_results = []
        for model_info in models_to_run:
            result = self._run_single_model(
                model_info, mode, platform, timeout, models_dir
            )
            all_results.append(result)

        # Aggregate metrics
        metrics = self._build_aggregate_metrics(all_results, mode)

        return {
            InfiniMetricsJson.RESULT_CODE: 0,
            InfiniMetricsJson.TIME: get_timestamp(),
            InfiniMetricsJson.RUN_ID: run_id,
            InfiniMetricsJson.TESTCASE: testcase,
            InfiniMetricsJson.CONFIG: config,
            InfiniMetricsJson.METRICS: metrics,
        }

    def _resolve_models(
        self, model_spec: str, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Resolve model specification to a list of model infos."""
        # Check config overrides first
        if config.get("model_name") == "all" or model_spec.lower() == "all":
            return get_all_models()

        if config.get("model_name"):
            model = get_model(config["model_name"])
            return [model] if model else []

        # Parse category/name format
        if "/" in model_spec:
            category, name = model_spec.split("/", 1)
            return get_models_by_category(category) if name.lower() == "all" else (
                [get_model(name)] if get_model(name) else []
            )

        # Try as category
        category_models = get_models_by_category(model_spec)
        if category_models:
            return category_models

        # Try as model name
        model = get_model(model_spec)
        return [model] if model else []

    def _run_single_model(
        self,
        model_info: Dict[str, Any],
        mode: str,
        platform: str,
        timeout: int,
        models_dir: Path,
    ) -> Dict[str, Any]:
        """Run a single model and collect results."""
        category = model_info["category"]
        name = model_info["name"]
        model_path = models_dir / model_info["path"]

        logger.info("Running model: %s/%s (mode=%s)", category, name, mode)

        result = {
            "category": category,
            "name": name,
            "mode": mode,
            "status": "unknown",
            "log": "",
            "parsed": None,
            "error": "",
        }

        # Determine script to run
        if mode == "train":
            script_name = model_info.get("train_script", "run_train.sh")
        else:
            script_name = model_info.get("eval_script", "run_eval.sh")
            if not script_name:
                script_name = "run_eval.sh"

        script_path = model_path / script_name
        if not script_path.exists():
            result["status"] = "skip"
            result["error"] = f"Script not found: {script_path}"
            logger.warning("Script not found: %s", script_path)
            return result

        # Build environment
        env = os.environ.copy()
        env["PLATFORM_ENV"] = get_platform_env_name(platform)

        # Add model-specific env vars
        for key, value in model_info.get("env_vars", {}).items():
            env[key] = value

        # Build command
        cmd = ["bash", str(script_name)]
        if env_args := model_info.get("env_args"):
            cmd.extend(env_args)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(model_path),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )

            log_text = proc.stdout + "\n" + proc.stderr
            result["log"] = log_text
            result["parsed"] = parse_log(log_text, name)

            if proc.returncode == 0:
                result["status"] = "pass"
            elif proc.returncode == 124:
                result["status"] = "timeout"
            else:
                result["status"] = "fail"
                result["error"] = proc.stderr[-500:] if proc.stderr else ""

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["error"] = f"Timed out after {timeout}s"
        except FileNotFoundError as e:
            result["status"] = "error"
            result["error"] = str(e)
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        logger.info(
            "Model %s/%s: status=%s", category, name, result["status"]
        )
        return result

    def _build_aggregate_metrics(
        self, results: List[Dict[str, Any]], mode: str
    ) -> List[Dict]:
        """Build aggregate metrics from all model results."""
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        skipped = sum(1 for r in results if r["status"] == "skip")
        timed_out = sum(1 for r in results if r["status"] == "timeout")
        errored = sum(1 for r in results if r["status"] == "error")

        metrics = [
            {
                "name": f"traditional.{mode}.total",
                "value": total,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": f"traditional.{mode}.passed",
                "value": passed,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": f"traditional.{mode}.failed",
                "value": failed,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": f"traditional.{mode}.skipped",
                "value": skipped,
                "type": "scalar",
                "unit": "",
            },
            {
                "name": f"traditional.{mode}.pass_rate",
                "value": round(passed / total * 100, 2) if total > 0 else 0,
                "type": "scalar",
                "unit": "%",
            },
        ]

        # Per-model details
        details = []
        for r in results:
            detail = {
                "name": f"{r['category']}/{r['name']}",
                "status": r["status"],
            }
            if r["parsed"]:
                detail["loss"] = r["parsed"]["final_loss"]
                detail["accuracy"] = r["parsed"]["final_accuracy"]
                detail["throughput"] = r["parsed"]["final_throughput"]
                detail["latency"] = r["parsed"]["final_latency"]
            if r["error"]:
                detail["error"] = r["error"][:500]
            details.append(detail)

        metrics.append(
            {
                "name": f"traditional.{mode}.details",
                "value": details,
                "type": "detail",
                "unit": "",
            }
        )

        return metrics

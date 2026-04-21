#!/usr/bin/env python3
"""Data source implementations for InfiniBench dashboard."""

import json
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_utils import (
    extract_accelerator_types,
    extract_run_info,
    load_summary_file,
    normalize_ci_summary,
    extract_failed_tests_details,
)

# Add project root to path for db module access (works regardless of cwd)
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Direct import to avoid triggering db/__init__.py (which imports pymongo)
import db.utils as _db_utils

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract data source for test results."""

    @abstractmethod
    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs."""
        pass

    @abstractmethod
    def load_test_result(self, identifier) -> Dict[str, Any]:
        """Load a single test result with full data."""
        pass

    @abstractmethod
    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summaries."""
        pass

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the data source type name."""
        pass


class FileDataSource(DataSource):
    """File-based data source (reads from JSON/CSV files)."""

    def __init__(self, results_dir: str = "./output"):
        self.results_dir = Path(results_dir)

    @property
    def source_type(self) -> str:
        return "file"

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs, filtering out summary files."""
        runs = []

        for json_file in self.results_dir.rglob("*.json"):
            try:
                if (
                    "summary" in json_file.name.lower()
                    or "dispatcher" in json_file.name.lower()
                ):
                    continue

                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not self._is_test_result_file(data):
                    continue

                testcase = data.get("testcase", "")
                if test_type and not testcase.startswith(test_type):
                    continue

                run_info = extract_run_info(data, json_file)
                run_info["accelerator_types"] = extract_accelerator_types(data)
                runs.append(run_info)

            except Exception as e:
                logger.debug(f"Skipping file {json_file}: {e}")

        runs.sort(key=lambda x: x["time"], reverse=True)
        return runs

    def load_test_result(self, json_path: Path) -> Dict[str, Any]:
        """Load a single test result with all data."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for metric in data.get("metrics", []):
            csv_url = metric.get("raw_data_url")
            if csv_url and not csv_url.startswith("http"):
                base_dir = _db_utils.get_csv_base_dir(data, json_path)
                csv_path = _db_utils.resolve_csv_path(csv_url, base_dir)

                if csv_path and csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        metric["data"] = df
                        metric["data_columns"] = list(df.columns)
                        metric["csv_path"] = str(csv_path)
                    except Exception as e:
                        logger.warning(f"Failed to load CSV {csv_path}: {e}")
                        metric["data"] = None
                else:
                    logger.debug(f"CSV not found: {csv_url} (base: {base_dir})")
                    metric["data"] = None

        return data

    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summary files from summary_output directory."""
        summary_dir = self.results_dir.parent / "summary_output"
        return load_summary_file(str(summary_dir))

    def load_ci_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Load the CI history from the summary file
        """
        summaries = self.load_summaries()
        enhanced_summaries = []

        for summary in summaries[:limit]:
            # Normalize CI metadata
            summary = normalize_ci_summary(summary)

            # Extract detailed information of failed test cases
            summary["failed_tests_details"] = extract_failed_tests_details(summary)

            # Add data source marker
            summary["_data_source"] = "file"
            summary["_summary_file"] = summary.get("file", "unknown")

            # Compute overall status
            total = summary.get("total_tests", 0)
            failed = summary.get("failed_tests", 0)
            if total == 0:
                summary["status"] = "无测试"
            elif failed == 0:
                summary["status"] = "成功"
            elif summary.get("successful_tests", 0) > 0:
                summary["status"] = "部分成功"
            else:
                summary["status"] = "失败"

            enhanced_summaries.append(summary)

        return enhanced_summaries

    def _is_test_result_file(self, data: Dict[str, Any]) -> bool:
        """Check if JSON file is a test result."""
        required = ["run_id", "testcase", "config"]
        return all(key in data for key in required) and "metrics" in data

    def _get_csv_base_dir(self, json_data: Dict[str, Any], json_path: Path) -> Path:
        """Get the correct base directory for CSV files."""
        config = json_data.get("config", {})
        output_dir = config.get("output_dir")

        if output_dir:
            output_path = Path(output_dir)
            if output_path.is_absolute():
                return output_path
            return json_path.parent / output_dir

        return json_path.parent

    def _resolve_csv_path(self, csv_url: str, base_dir: Path) -> Optional[Path]:
        """Resolve CSV path from raw_data_url."""
        try:
            if not csv_url:
                return None

            if csv_url.startswith("./"):
                csv_url = csv_url[2:]

            # Try a variety of possible paths
            candidates = [
                base_dir / csv_url,
                base_dir / Path(csv_url).name,
                base_dir.parent / csv_url,
                base_dir.parent / Path(csv_url).name,
            ]

            for p in candidates:
                if p.exists():
                    return p

            return None
        except Exception:
            return None

#!/usr/bin/env python3
"""MongoDB data source for InfiniBench dashboard."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_sources import DataSource
from .data_utils import (
    extract_accelerator_types,
    extract_run_info,
    normalize_ci_summary,
    extract_failed_tests_details,
)

logger = logging.getLogger(__name__)


class MongoDataSource(DataSource):
    """MongoDB-based data source."""

    def __init__(self, config=None):
        self._config = config
        self._client = None
        self._repository = None
        self._connected = False

    def _connect(self):
        """Lazy connection to MongoDB."""
        if self._connected:
            return self._connected

        try:
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from db import MongoDBClient, TestRunRepository, DispatcherSummaryRepository

            if self._config:
                self._client = MongoDBClient(self._config)
            else:
                self._client = MongoDBClient()

            if self._client.health_check():
                from db.config import DatabaseConfig

                config = self._config or DatabaseConfig.from_env()
                self._repository = TestRunRepository(
                    self._client.get_collection(config.collection_name)
                )

                self._summary_repo = DispatcherSummaryRepository(
                    self._client.get_collection(config.summary_collection_name)
                )
                self._connected = True
                logger.info("Connected to MongoDB data source")
            else:
                logger.warning("MongoDB health check failed")

        except Exception as e:
            logger.warning(f"Failed to connect to MongoDB: {e}")
            self._connected = False

        return self._connected

    @property
    def source_type(self) -> str:
        return "mongodb"

    def is_connected(self) -> bool:
        """Check if MongoDB is connected."""
        return self._connected or self._connect()

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs from MongoDB."""
        if not self._connect():
            logger.warning("MongoDB not connected, returning empty list")
            return []

        runs = self._repository.list_test_runs(test_type=test_type)
        result = []

        for run in runs:
            run_info = extract_run_info(run)
            run_info["accelerator_types"] = extract_accelerator_types(run)
            result.append(run_info)

        result.sort(key=lambda x: x["time"], reverse=True)
        return result

    def load_test_result(self, run_id: str) -> Dict[str, Any]:
        """Load a single test result with full data from MongoDB."""
        if not self._connect():
            logger.warning("MongoDB not connected")
            return {}

        data = self._repository.find_by_run_id(run_id)
        if not data:
            return {}

        for metric in data.get("metrics", []):
            if "data" in metric and isinstance(metric["data"], list):
                if metric["data"]:
                    metric["data"] = pd.DataFrame(metric["data"])
                    if "data_columns" not in metric:
                        metric["data_columns"] = list(metric["data"].columns)

        data.pop("_id", None)
        data.pop("_metadata", None)

        return data

    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summaries from MongoDB."""
        if not self._connect():
            logger.warning("MongoDB not connected, returning empty list")
            return []

        try:
            from db import DispatcherSummaryRepository
            from db.config import DatabaseConfig

            config = self._config or DatabaseConfig.from_env()
            summary_collection = self._client.get_collection(
                config.summary_collection_name
            )
            summary_repo = DispatcherSummaryRepository(summary_collection)
            summaries = summary_repo.list_summaries()

            for s in summaries:
                s.pop("_id", None)
                s.pop("_metadata", None)

            return summaries
        except Exception as e:
            logger.warning(f"Failed to load summaries from MongoDB: {e}")
            return []

    def load_ci_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Load CI history with enhanced information from MongoDB.
        """
        if not self._connect():
            logger.warning("MongoDB not connected, returning empty list")
            return []

        try:
            # Retrieve CI summaries from the summary repository
            summaries = self._summary_repo.list_summaries(limit=limit)
            enhanced_summaries = []

            for summary in summaries:
                # Remove internal MongoDB fields
                summary.pop("_id", None)
                summary.pop("_metadata", None)

                # Normalize CI summary format
                summary = normalize_ci_summary(summary)

                # Extract failed test details
                if "failed_tests_details" not in summary:
                    # Try to load failure details from associated test results
                    failed_details = self._load_failed_tests_for_summary(summary)
                    summary["failed_tests_details"] = failed_details

                # Add data source marker
                summary["_data_source"] = "mongodb"

                # Derive overall status
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

        except Exception as e:
            logger.warning(f"Failed to load CI history from MongoDB: {e}")
            return []

    def _load_failed_tests_for_summary(
        self, summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Load failed test details for a given summary.
        """
        failed_details = []

        # If run_ids are available, try loading each failed test individually
        if "run_ids" in summary:
            for run_id in summary.get("run_ids", []):
                try:
                    test_result = self.load_test_result(run_id)
                    if test_result and not test_result.get("success", True):
                        failed_details.append(
                            {
                                "test_name": test_result.get("testcase", "unknown"),
                                "run_id": run_id,
                                "error": test_result.get("error_msg", "Unknown error"),
                                "duration": self._extract_duration(test_result),
                                "logs": test_result.get("logs", ""),
                                "config": test_result.get("config", {}),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Failed to load test {run_id}: {e}")

        # If test_results are already embedded in the summary
        elif "test_results" in summary:
            for test in summary["test_results"]:
                if not test.get("success", True):
                    failed_details.append(
                        {
                            "test_name": test.get(
                                "name", test.get("testcase", "unknown")
                            ),
                            "error": test.get(
                                "error", test.get("message", "Unknown error")
                            ),
                            "duration": test.get("duration", 0),
                            "logs": test.get("logs", test.get("output", "")),
                        }
                    )

        return failed_details

    def _extract_duration(self, test_result: Dict[str, Any]) -> float:
        """Extract duration from test result metrics."""
        for metric in test_result.get("metrics", []):
            if metric.get("name") == "duration":
                return metric.get("value", 0)
        return 0

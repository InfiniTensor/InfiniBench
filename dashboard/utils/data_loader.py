#!/usr/bin/env python3
"""Unified data loader for InfiniBench dashboard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .mongo_data_source import MongoDataSource

from .data_sources import DataSource, FileDataSource
from .data_utils import extract_accelerator_types, extract_run_info, get_friendly_size

logger = logging.getLogger(__name__)


class InfiniBenchDataLoader:
    """
    Unified data loader supporting multiple sources.

    Supports:
    - File-based loading (default)
    - MongoDB-based loading
    - Automatic fallback from MongoDB to files
    """

    def __init__(
        self,
        results_dir: str = "./output",
        use_mongodb: bool = False,
        mongo_config=None,
        fallback_to_files: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            results_dir: Directory containing test result files
            use_mongodb: If True, use MongoDB as primary data source
            mongo_config: Optional MongoDB configuration
            fallback_to_files: If True, fall back to file loading if MongoDB fails
        """
        self.results_dir = Path(results_dir)
        self._fallback_to_files = fallback_to_files
        self._use_mongodb = use_mongodb
        self._mongo_config = mongo_config
        self._source: Optional[DataSource] = None

        if use_mongodb:
            self._init_mongodb_source()
        else:
            self._source = FileDataSource(results_dir)

    def _try_connect_mongo(self) -> Optional[MongoDataSource]:
        """
        Try to connect to MongoDB.

        Returns:
            MongoDataSource if connected, None otherwise
        """
        try:
            from .mongo_data_source import MongoDataSource

            mongo_source = MongoDataSource(self._mongo_config)
            if mongo_source.is_connected():
                return mongo_source
        except ImportError as e:
            logger.warning(f"MongoDB dependencies not installed ({e})")
        return None

    def _apply_mongo_or_fallback(self, mongo_source: Optional[MongoDataSource]):
        """Apply MongoDB source or fallback to files based on configuration."""
        if mongo_source:
            self._source = mongo_source
            self._use_mongodb = True
        elif self._fallback_to_files:
            logger.warning("MongoDB unavailable, using file-based loading")
            self._source = FileDataSource(str(self.results_dir))
            self._use_mongodb = False
        else:
            raise RuntimeError(
                "MongoDB connection failed and fallback is disabled. "
                "Install pymongo to use MongoDB."
            )

    def _init_mongodb_source(self):
        """Initialize MongoDB data source with optional fallback."""
        mongo_source = self._try_connect_mongo()
        self._apply_mongo_or_fallback(mongo_source)

    @property
    def source_type(self) -> str:
        """Get the current data source type."""
        return self._source.source_type if self._source else "none"

    @property
    def is_connected(self) -> bool:
        """Check if data source is available."""
        return self._source is not None

    @property
    def is_using_mongodb(self) -> bool:
        """Check if currently using MongoDB."""
        return (
            self._use_mongodb and self._source and self._source.source_type == "mongodb"
        )

    def switch_to_mongodb(self, mongo_config=None) -> bool:
        """
        Switch to MongoDB data source.

        Returns:
            True if switch was successful
        """
        if mongo_config:
            self._mongo_config = mongo_config

        mongo_source = self._try_connect_mongo()
        if mongo_source:
            self._source = mongo_source
            self._use_mongodb = True
            return True
        else:
            logger.warning("Failed to switch to MongoDB, keeping current source")
            return False

    def switch_to_files(self, results_dir: str = None):
        """Switch to file-based data source."""
        if results_dir:
            self.results_dir = Path(results_dir)
        self._source = FileDataSource(str(self.results_dir))
        self._use_mongodb = False

    def list_test_runs(self, test_type: str = None) -> List[Dict[str, Any]]:
        """List all test runs."""
        if self._source is None:
            return []
        return self._source.list_test_runs(test_type)

    def load_test_result(self, identifier) -> Dict[str, Any]:
        """
        Load a single test result with all data.

        Args:
            identifier: For file source, a Path to JSON file.
                       For MongoDB source, a run_id string.
        """
        if self._source is None:
            return {}
        return self._source.load_test_result(identifier)

    def load_summaries(self) -> List[Dict[str, Any]]:
        """Load dispatcher summaries from the current data source."""
        if self._source is None:
            return []
        return self._source.load_summaries()

    def load_ci_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Load CI history with detailed execution information.

        Args:
            limit: Maximum number of CI runs to load

        Returns:
            List of CI run summaries with enhanced information
        """
        if self._source is None:
            return []

        # Check if the source has load_ci_history method
        if hasattr(self._source, "load_ci_history"):
            return self._source.load_ci_history(limit)

        # Fallback: use load_summaries and enhance them
        summaries = self._source.load_summaries()
        return self._enhance_summaries(summaries[:limit])

    def _enhance_summaries(
        self, summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enhance summaries with CI information and failure details.
        This is a fallback method when the data source doesn't provide enhanced data.
        """
        enhanced = []
        for summary in summaries:
            # Normalize fields
            enhanced_summary = summary.copy()

            # Set default CI fields with placeholder markers
            ci_fields = [
                "commit",
                "branch",
                "commit_message",
                "author",
                "pipeline_id",
                "ci_url",
            ]
            for field in ci_fields:
                if field not in enhanced_summary:
                    enhanced_summary[field] = f"unknown_{field}"
                    enhanced_summary[f"{field}_placeholder"] = True

            # Calculate status
            total = enhanced_summary.get("total_tests", 0)
            failed = enhanced_summary.get("failed_tests", 0)
            if total == 0:
                enhanced_summary["status"] = "无测试"
            elif failed == 0:
                enhanced_summary["status"] = "成功"
            elif enhanced_summary.get("successful_tests", 0) > 0:
                enhanced_summary["status"] = "部分成功"
            else:
                enhanced_summary["status"] = "失败"

            # Add failed tests details if not present
            if "failed_tests_details" not in enhanced_summary:
                enhanced_summary["failed_tests_details"] = []

            # Add data source marker
            enhanced_summary["_data_source"] = self.source_type

            enhanced.append(enhanced_summary)

        return enhanced


# Re-export from sibling modules
from .data_sources import DataSource, FileDataSource
from .data_utils import (
    get_friendly_size,
    extract_accelerator_types,
    extract_run_info,
)

__all__ = [
    "InfiniBenchDataLoader",
    "DataSource",
    "FileDataSource",
    "get_friendly_size",
    "extract_accelerator_types",
    "extract_run_info",
]

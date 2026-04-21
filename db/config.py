#!/usr/bin/env python3
"""MongoDB configuration for InfiniBench."""

from dataclasses import dataclass
import os


@dataclass
class DatabaseConfig:
    """MongoDB configuration settings."""

    mongo_uri: str = "mongodb://localhost:27017"
    database_name: str = "infinibench"
    collection_name: str = "test_runs"
    summary_collection_name: str = "dispatcher_summaries"
    connection_timeout_ms: int = 5000
    max_pool_size: int = 10

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        return cls(
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            database_name=os.getenv("MONGO_DB_NAME", "infinibench"),
            collection_name=os.getenv("MONGO_COLLECTION", "test_runs"),
            summary_collection_name=os.getenv(
                "MONGO_SUMMARY_COLLECTION", "dispatcher_summaries"
            ),
            connection_timeout_ms=int(os.getenv("MONGO_TIMEOUT_MS", "5000")),
            max_pool_size=int(os.getenv("MONGO_POOL_SIZE", "10")),
        )

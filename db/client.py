#!/usr/bin/env python3
"""MongoDB connection management for InfiniBench."""

import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database

from .config import DatabaseConfig

logger = logging.getLogger(__name__)


class MongoDBConnectionError(Exception):
    """Raised when MongoDB connection fails."""

    pass


class MongoDBClient:
    """Singleton MongoDB connection manager."""

    _instance: Optional["MongoDBClient"] = None
    _client: Optional[MongoClient] = None

    def __new__(cls, config: Optional[DatabaseConfig] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[DatabaseConfig] = None):
        if self._client is None:
            self._config = config or DatabaseConfig.from_env()
            self._connect()

    def _connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            self._client = MongoClient(
                self._config.mongo_uri,
                serverSelectionTimeoutMS=self._config.connection_timeout_ms,
                maxPoolSize=self._config.max_pool_size,
            )
            # Test connection
            self._client.admin.command("ping")
            logger.info(f"Connected to MongoDB at {self._config.mongo_uri}")
        except Exception as e:
            self._client = None
            raise MongoDBConnectionError(
                f"Failed to connect to MongoDB at {self._config.mongo_uri}: {e}"
            )

    @property
    def database(self) -> Database:
        """Get the database instance."""
        if self._client is None:
            raise MongoDBConnectionError("MongoDB client not connected")
        return self._client[self._config.database_name]

    def get_collection(self, name: str = None):
        """Get a collection by name."""
        collection_name = name or self._config.collection_name
        return self.database[collection_name]

    def health_check(self) -> bool:
        """Check if connection is healthy."""
        try:
            if self._client is None:
                return False
            self._client.admin.command("ping")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the connection."""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("MongoDB connection closed")

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance:
            cls._instance.close()
        cls._instance = None
        cls._client = None

#!/usr/bin/env python3
"""
InfiniBench MongoDB integration module.

This module provides:
- MongoDB connection management (client.py)
- Database configuration (config.py)
- Test result repository (repository.py)
- Data import from JSON/CSV files (importer.py)
- File watcher for auto-import (watcher.py)

Usage:
    from db import MongoDBClient, DataImporter

    # Connect to MongoDB
    client = MongoDBClient()
    repo = TestRunRepository(client.get_collection("test_runs"))

    # Import existing results
    importer = DataImporter(repo)
    importer.import_directory(Path("./output"))

    # Or use the watcher for auto-import
    from db.watcher import Watcher
    watcher = Watcher(output_dir=Path("./output"))
    watcher.run_forever()

CLI Usage:
    # Start watching for new files
    python -m db.watcher

    # One-time scan
    python -m db.watcher --scan
"""

# Conditionally import MongoDB-dependent modules
# This allows db.utils to be imported without pymongo
try:
    from .client import MongoDBClient, MongoDBConnectionError
    from .config import DatabaseConfig
    from .importer import DataImporter
    from .repository import DispatcherSummaryRepository, TestRunRepository

    __all__ = [
        "DatabaseConfig",
        "MongoDBClient",
        "MongoDBConnectionError",
        "TestRunRepository",
        "DispatcherSummaryRepository",
        "DataImporter",
    ]
except ImportError:
    # pymongo not installed - MongoDB features unavailable
    __all__ = []

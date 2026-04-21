#!/usr/bin/env python3
"""Time-related utility functions."""

from datetime import datetime


def get_timestamp() -> str:
    """
    Get current timestamp in standardized format.

    Returns:
        Timestamp string in 'YYYY-MM-DD HH:MM:SS' format
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

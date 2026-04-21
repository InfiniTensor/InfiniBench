#!/usr/bin/env python3
"""
Path/filename utility helpers.
"""

import re
from typing import Any


def sanitize_filename(name: Any) -> str:
    """
    Make name safe to be used as a filename.
    Keep letters/numbers/_/./- ; replace others with '_'.
    """
    if name is None:
        return "unknown"
    s = str(name).strip()
    if not s:
        return "unknown"
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s or "unknown"

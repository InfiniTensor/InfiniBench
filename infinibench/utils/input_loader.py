#!/usr/bin/env python3
"""Input loading utilities for test specifications."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_input_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load input from single file, returns list of test inputs."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def load_inputs_from_paths(input_paths: List[str]) -> List[Dict[str, Any]]:
    """Load all test inputs from files and directories."""
    all_inputs = []

    for path_str in input_paths:
        path = Path(path_str)
        try:
            if path.is_file():
                inputs = load_input_file(path)
                all_inputs.extend(inputs)
                logger.info(f"Loaded {len(inputs)} input(s) from {path_str}")
            elif path.is_dir():
                for json_file in sorted(path.glob("*.json")):
                    inputs = load_input_file(json_file)
                    all_inputs.extend(inputs)
                logger.info(
                    f"Loaded {len(list(path.glob('*.json')))} file(s) from {path_str}"
                )
            else:
                logger.warning(f"Path not found: {path_str}")
        except Exception as e:
            logger.error(f"Error loading {path_str}: {e}")

    return all_inputs

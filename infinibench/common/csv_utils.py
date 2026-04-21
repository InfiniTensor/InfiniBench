#!/usr/bin/env python3
"""CSV Utilities - Common CSV file operations"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


def save_csv(
    output_dir: Path,
    filename: str,
    data: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    add_index: bool = True,
    remove_internal_fields: Optional[List[str]] = None,
) -> Path:
    """
    Save data to CSV file with optional sorting and index column.

    Args:
        output_dir: Directory to save the CSV file
        filename: Name of the CSV file
        data: List of dictionaries containing the data
        fieldnames: List of field names (auto-detected if None)
        sort_by: Field name to sort by (supports special handling for _sort_key)
        add_index: Whether to add an index column (default: True)
        remove_internal_fields: List of internal field names to remove (default: ["_sort_key"])

    Returns:
        Path to the created CSV file
    """
    if remove_internal_fields is None:
        remove_internal_fields = ["_sort_key"]

    csv_path = output_dir / filename

    # Auto-detect fieldnames if not provided
    if not fieldnames:
        fieldnames = sorted(set().union(*data))

    # Prepare data with optional index
    data_with_index = (
        [{"index": idx, **row} for idx, row in enumerate(data)] if add_index else data
    )

    # Sort if needed
    if sort_by:
        data_with_index = _sort_data(data_with_index, sort_by)

    # Remove internal fields
    final_data = [
        {
            k: v
            for k, v in row.items()
            if k not in remove_internal_fields or k == sort_by
        }
        for row in data_with_index
    ]

    # Prepare final fieldnames
    if add_index:
        fieldnames = ["index"] + [
            f for f in fieldnames if f not in ("index", *remove_internal_fields)
        ]
    else:
        fieldnames = [f for f in fieldnames if f not in remove_internal_fields]

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_data)

    logger.info("CSV saved: %s", csv_path)
    return csv_path


def _sort_data(data: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    """
    Sort data by specified field with special handling for common patterns.

    Args:
        data: List of dictionaries to sort
        sort_by: Field name to sort by

    Returns:
        Sorted list of dictionaries
    """

    def sort_key(x: Dict[str, Any]) -> float:
        # Handle explicit sort key
        if "_sort_key" in x:
            return float(x["_sort_key"])

        # Get the value to sort by
        val = x.get(sort_by, "0")

        # Handle common size unit patterns (kB, MB, GB)
        if isinstance(val, str):
            val = val.replace(" kB", "").replace(" MB", "").replace(" GB", "")

        return float(str(val))

    return sorted(data, key=sort_key)


def create_timeseries_metric(
    output_dir: Path,
    metric_name: str,
    data: List[Dict[str, Any]],
    base_filename: str,
    fields: List[str],
    unit: str,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a timeseries metric with associated CSV file.

    Args:
        output_dir: Directory to save the CSV file
        metric_name: Name of the metric
        data: Data points for the timeseries
        base_filename: Base name for the CSV file (timestamp will be appended)
        fields: Field names for the CSV
        unit: Unit of measurement
        timestamp: Optional timestamp string (auto-generated if None)

    Returns:
        Dictionary representing the timeseries metric
    """

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = f"{base_filename}_{timestamp}.csv"
    csv_file = save_csv(
        output_dir, csv_filename, data, fieldnames=fields, sort_by=fields[0]
    )

    return {
        "name": metric_name,
        "type": "timeseries",
        "raw_data_url": f"./{csv_file.name}",
        "unit": unit,
    }

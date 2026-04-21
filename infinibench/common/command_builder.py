#!/usr/bin/env python3
"""Command Builder - Unified command construction utilities"""

from typing import Dict, Any, List, Tuple


def build_command_from_config(
    base_command: List[str],
    config: Dict[str, Any],
    param_mappings: List[Tuple[str, str]],
) -> List[str]:
    """
    Build command from base command and configuration parameters.

    This function takes a base command and adds optional parameters from config
    based on the provided mappings.

    Args:
        base_command: Base command as list of strings (e.g., ['./program', '--flag'])
        config: Configuration dictionary containing parameters
        param_mappings: List of (config_key, param_name) tuples
                       config_key: key to look up in config dictionary
                       param_name: command-line parameter name (e.g., '--device')

    Returns:
        Complete command as list of strings

    Example:
        >>> cmd = build_command_from_config(
        ...     ['./benchmark', '--test'],
        ...     {'device_id': 0, 'iterations': 100},
        ...     [('device_id', '--device'), ('iterations', '--iterations')]
        ... )
        >>> cmd
        ['./benchmark', '--test', '--device', '0', '--iterations', '100']
    """
    cmd = base_command.copy()

    # Add parameters that exist in config
    for config_key, param_name in param_mappings:
        if config_key in config:
            cmd.extend([param_name, str(config[config_key])])

    return cmd

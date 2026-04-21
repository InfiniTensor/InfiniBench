#!/usr/bin/env python3
"""Base Adapter - Unified Interface for All Test Types"""

import abc
import logging
from typing import Dict, Any, Union, Optional

from infinibench.input import TestInput
from infinibench.common.constants import InfiniBenchJson
from infinibench.utils.time_utils import get_timestamp

logger = logging.getLogger(__name__)


class BaseAdapter(abc.ABC):
    """
    Unified adapter base class for all test types (inference, operator, training).

    Lifecycle:
        1. __init__() - Create adapter (in Dispatcher)
        2. setup(config) - Initialize resources (in Executor)
        3. process(test_input) - Execute test
        4. teardown() - Cleanup resources
    """

    @abc.abstractmethod
    def process(self, test_input: Union[TestInput, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute the test.

        Args:
            test_input: TestInput object or dict with testcase, config, etc.

        Returns:
            Dict with:
                - 'result_code': int (0 = success, non-zero = error code)
                - 'metrics': list (performance metrics)
                - ...
        """
        pass

    def setup(self, config: Dict[str, Any]) -> None:
        """
        Initialize resources before running tests.

        Called by Executor before process().

        Args:
            config: Configuration dict from test_input.config
        """
        pass

    def teardown(self) -> None:
        """
        Cleanup resources after tests complete.

        Called by Executor after process() (even if process() fails).
        """
        pass

    def _normalize_test_input(
        self, test_input: Union[TestInput, Dict[str, Any], Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize test input to dictionary format.

        Args:
            test_input: TestInput object, dict, or other type

        Returns:
            Dict representation of test_input, or None if conversion fails
        """
        if isinstance(test_input, dict):
            return test_input
        if hasattr(test_input, "to_dict"):
            return test_input.to_dict()
        return None

    def _create_error_response(
        self,
        error_msg: str,
        test_input: Optional[Dict[str, Any]] = None,
        result_code: int = 1,
    ) -> Dict[str, Any]:
        """
        Create standardized error response.

        Args:
            error_msg: Error message describing the failure
            test_input: Optional original test input for context
            result_code: Error code (default: 1)

        Returns:
            Dict with error information in standardized format
        """
        response = {
            InfiniBenchJson.RESULT_CODE: result_code,
            InfiniBenchJson.TIME: get_timestamp(),
            InfiniBenchJson.ERROR_MSG: error_msg,
            InfiniBenchJson.METRICS: [],
        }

        # Include context if test_input is provided
        if test_input:
            response.update(
                {
                    InfiniBenchJson.RUN_ID: test_input.get(InfiniBenchJson.RUN_ID, ""),
                    InfiniBenchJson.TESTCASE: test_input.get(
                        InfiniBenchJson.TESTCASE, ""
                    ),
                    InfiniBenchJson.CONFIG: test_input.get(InfiniBenchJson.CONFIG, {}),
                }
            )

        return response

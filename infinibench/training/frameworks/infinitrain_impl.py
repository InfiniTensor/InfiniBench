import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class InfinitrainImpl:
    """InfiniTrain training implementation (placeholder - not implemented)."""

    def __init__(
        self,
        config: Dict[str, Any],
        resolved_device_count: int,
        run_id: Optional[str] = None,
    ):
        self.config = config
        self.resolved_device_count = resolved_device_count
        self.run_id = run_id if run_id is not None else config.get("run_id", "unknown")
        self.logger = logger
        self.logger.warning(
            f"InfiniTrain implementation is not ready yet, run_id={self.run_id}"
        )

    def run(self) -> Dict[str, Any]:
        """Placeholder implementation."""
        raise NotImplementedError(
            "InfiniTrain implementation is not ready yet. "
            "Please implement when InfiniTrain is available."
        )

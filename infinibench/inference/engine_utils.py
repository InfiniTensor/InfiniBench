#!/usr/bin/env python3
"""
Shared engine utilities for inference engines.
"""

from typing import Any


class AcceleratorMonitorMixin:
    """
    Mixin for accelerator monitoring lifecycle used by inference engines.

    Requires:
      - self.config
      - self.result_data (dict)
    """

    monitor = None

    def _start_accelerator_monitor(self):
        dev = getattr(self.config, "device", None)

        if isinstance(dev, dict):
            accelerator = dev.get("accelerator", "nvidia")
            device_ids = dev.get("device_ids", None)
            cpu_only = dev.get("cpu_only", False)
        else:
            accelerator = getattr(getattr(dev, "accelerator", None), "value", "nvidia")
            device_ids = getattr(dev, "device_ids", None)
            cpu_only = getattr(dev, "cpu_only", False)

        if cpu_only:
            return

        from infinibench.utils.accelerator_monitor import create_accelerator_monitor

        self.monitor = create_accelerator_monitor(
            accelerator_type=accelerator,
            device_ids=device_ids,
        )
        self.monitor.start_monitoring()

    def _stop_and_collect_monitor(self):
        if not self.monitor:
            return

        self.monitor.stop_monitoring()
        peak_gb = self.monitor.get_peak_memory_gb()

        if hasattr(self, "result_data") and isinstance(self.result_data, dict):
            self.result_data["peak_memory_usage"] = peak_gb

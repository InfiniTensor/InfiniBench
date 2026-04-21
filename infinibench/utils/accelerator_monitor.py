"""
Accelerator Monitoring Module
"""

import random
import subprocess
import threading
import time
from abc import ABC, abstractmethod
import logging
from enum import Enum
from typing import Dict, List, Optional, Union
from infinibench.common.constants import ProcessorType, AcceleratorType

logger = logging.getLogger(__name__)


class AcceleratorMonitor(ABC):
    """Abstract base class for accelerator monitoring"""

    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        accelerator_type: AcceleratorType = None,
    ):
        """
        Initialize accelerator monitor

        Args:
            device_ids: List of device IDs to monitor. If None, monitor all devices.
            accelerator_type: Type of accelerator being monitored
        """
        self.device_ids = device_ids
        self.accelerator_type = accelerator_type
        self.peak_memory_mib = 0  # Peak memory usage (MiB)
        self.monitor_thread = None
        self._stop_monitoring_flag = False
        self.poll_interval = 0.5  # Polling interval (seconds)

        logger.info(
            f"AcceleratorMonitor initialized for {accelerator_type.value}, devices: {device_ids}"
        )

    @abstractmethod
    def get_current_memory_usage(self) -> List[int]:
        """Get current accelerator memory usage for all devices (in MiB)"""
        pass

    @abstractmethod
    def get_peak_memory_allocated(self) -> Optional[int]:
        """Get peak memory allocated via framework APIs (in bytes)"""
        pass

    def start_monitoring(self):
        """Start monitoring accelerator memory usage"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring already started")
            return

        self._stop_monitoring_flag = False
        self.peak_memory_mib = 0

        def monitor_loop():
            logger.info(f"{self.accelerator_type.value} monitoring started")
            while not self._stop_monitoring_flag:
                try:
                    current_mem = self.get_current_memory_usage()
                    if current_mem:
                        current_peak = max(current_mem)
                        if current_peak > self.peak_memory_mib:
                            self.peak_memory_mib = current_peak
                            logger.debug(f"New peak memory: {self.peak_memory_mib} MiB")
                except Exception as e:
                    logger.debug(f"Error getting accelerator memory: {e}")

                time.sleep(self.poll_interval)

            logger.info(f"{self.accelerator_type.value} monitoring stopped")

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring memory usage"""
        self._stop_monitoring_flag = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop gracefully")

    def get_peak_memory_gb(self) -> float:
        """Get peak memory usage in GB"""
        return (
            round(self.peak_memory_mib / 1024.0, 6) if self.peak_memory_mib > 0 else 0.0
        )

    def get_peak_memory_mib(self) -> int:
        """Get peak memory usage in MiB"""
        return self.peak_memory_mib

    def get_device_count(self) -> int:
        """Get number of devices being monitored"""
        if self.device_ids:
            return len(self.device_ids)
        return 1  # Default to 1 if not specified


class NVIDIAAcceleratorMonitor(AcceleratorMonitor):
    """NVIDIA accelerator monitor implementation"""

    def __init__(self, device_ids=None):
        super().__init__(device_ids, AcceleratorType.NVIDIA)

    def get_current_memory_usage(self) -> List[int]:
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]

            if self.device_ids:
                device_str = ",".join(str(d) for d in self.device_ids)
                cmd.extend(["--id", device_str])

            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, timeout=5
            )

            lines = [l.strip() for l in out.splitlines() if l.strip()]
            memory_values = [int(x) for x in lines if x.isdigit()]

            if memory_values:
                logger.debug(f"Current NVIDIA GPU memory usage: {memory_values} MiB")
                return memory_values
            else:
                return []

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            logger.debug(f"nvidia-smi command failed or not available: {e}")
            return self._try_pytorch_fallback()
        except Exception as e:
            logger.debug(f"Error getting NVIDIA GPU memory: {e}")
            return []

    def get_peak_memory_allocated(self) -> Optional[int]:
        """Get peak memory via PyTorch CUDA API (in bytes)"""
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                max_memory = 0
                for i in range(torch.cuda.device_count()):
                    max_memory = max(max_memory, torch.cuda.max_memory_allocated(i))
                return max_memory
        except ImportError:
            logger.debug("PyTorch not available for NVIDIA peak memory")
        return None

    def _try_pytorch_fallback(self) -> List[int]:
        """Fallback to PyTorch CUDA memory info if nvidia-smi fails"""
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                memory_values = []
                for i in range(torch.cuda.device_count()):
                    # Convert bytes to MiB
                    mem_mib = torch.cuda.memory_allocated(i) // (1024 * 1024)
                    memory_values.append(mem_mib)
                logger.debug(
                    f"Using PyTorch fallback for NVIDIA memory: {memory_values} MiB"
                )
                return memory_values
        except ImportError:
            pass
        return []


class MLUAcceleratorMonitor(AcceleratorMonitor):
    """Cambricon MLU accelerator monitor implementation"""

    def __init__(self, device_ids=None):
        super().__init__(device_ids, AcceleratorType.CAMBRICON_MLU)

    def get_current_memory_usage(self) -> List[int]:
        try:
            # Cambricon MLU typically uses cnmon or similar tools
            # This is an example - actual command may vary
            cmd = ["cnmon", "info", "-m"]

            out = subprocess.check_output(
                cmd, text=True, stderr=subprocess.DEVNULL, timeout=5
            )

            # Parse cnmon output to extract memory usage
            # This is a simplified example - actual parsing would be more complex
            lines = out.splitlines()
            memory_values = []

            for line in lines:
                if "Memory Used" in line or "MLU Memory" in line:
                    # Example parsing logic
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "MB" in part or "MiB" in part:
                            try:
                                mem_value = int(parts[i - 1])
                                memory_values.append(mem_value)
                            except (ValueError, IndexError):
                                pass

            if memory_values:
                logger.debug(f"Current MLU memory usage: {memory_values} MiB")
                return memory_values

            # If parsing fails, try PyTorch MLU API
            return self._try_pytorch_mlu_fallback()

        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
        ) as e:
            logger.debug(f"MLU monitoring command failed: {e}")
            return self._try_pytorch_mlu_fallback()
        except Exception as e:
            logger.debug(f"Error getting MLU memory: {e}")
            return []

    def get_peak_memory_allocated(self) -> Optional[int]:
        """Get peak memory via PyTorch MLU API if available"""
        try:
            import torch

            if hasattr(torch, "mlu") and torch.mlu.is_available():
                # PyTorch MLU API may differ - this is an example
                if hasattr(torch.mlu, "max_memory_allocated"):
                    max_memory = 0
                    for i in range(torch.mlu.device_count()):
                        max_memory = max(max_memory, torch.mlu.max_memory_allocated(i))
                    return max_memory
        except (ImportError, AttributeError):
            logger.debug("PyTorch MLU not available for peak memory")
        return None

    def _try_pytorch_mlu_fallback(self) -> List[int]:
        """Fallback to PyTorch MLU memory info"""
        try:
            import torch

            if hasattr(torch, "mlu") and torch.mlu.is_available():
                memory_values = []
                for i in range(torch.mlu.device_count()):
                    # MLU might have different API
                    if hasattr(torch.mlu, "memory_allocated"):
                        mem_bytes = torch.mlu.memory_allocated(i)
                        mem_mib = mem_bytes // (1024 * 1024)
                        memory_values.append(mem_mib)
                logger.debug(f"Using PyTorch MLU fallback: {memory_values} MiB")
                return memory_values
        except (ImportError, AttributeError):
            pass
        return []


class MockAcceleratorMonitor(AcceleratorMonitor):
    """Mock accelerator monitor for testing"""

    def __init__(self, device_ids=None):
        super().__init__(device_ids, AcceleratorType.MOCK)

    def get_current_memory_usage(self) -> List[int]:
        """Return mock memory usage data"""
        if self.device_ids:
            return [random.randint(100, 1000) for _ in self.device_ids]
        else:
            return [random.randint(100, 1000)]

    def get_peak_memory_allocated(self) -> Optional[int]:
        """Return mock peak memory"""
        return random.randint(512 * 1024 * 1024, 2048 * 1024 * 1024)  # 512MB-2GB


class GenericAcceleratorMonitor(AcceleratorMonitor):
    """Generic accelerator monitor for unsupported/unknown accelerator types"""

    def __init__(self, device_ids=None, accel_type="generic"):
        # Use GENERIC enum value for generic monitors
        super().__init__(device_ids, AcceleratorType.GENERIC)
        self.accel_type = accel_type

    def get_current_memory_usage(self) -> List[int]:
        """Return generic/placeholder memory usage data"""
        logger.warning(
            f"GenericAcceleratorMonitor ({self.accel_type}): "
            "Returning placeholder memory usage data"
        )
        if self.device_ids:
            return [0] * len(self.device_ids)
        return [0]

    def get_peak_memory_allocated(self) -> Optional[int]:
        """Return generic/placeholder peak memory"""
        logger.warning(
            f"GenericAcceleratorMonitor ({self.accel_type}): "
            "Returning placeholder peak memory"
        )
        return 0


def create_accelerator_monitor(
    accelerator_type: Union[str, AcceleratorType] = "nvidia",
    device_ids: Optional[List[int]] = None,
    fallback_to_generic: bool = True,
) -> AcceleratorMonitor:
    """
    Factory function to create accelerator monitor

    Args:
        accelerator_type: Accelerator platform name or enum
        device_ids: List of device IDs to monitor
        fallback_to_generic: If True, fallback to generic monitor when specific one fails

    Returns:
        Instance of AcceleratorMonitor
    """
    if isinstance(accelerator_type, str):
        accelerator_type = accelerator_type.lower()
        # Map string to enum
        type_map = {
            "nvidia": AcceleratorType.NVIDIA,
            "amd": AcceleratorType.AMD,
            "ascend": AcceleratorType.ASCEND,
            "cambricon": AcceleratorType.CAMBRICON,
        }

        accel_enum = type_map.get(accelerator_type)
        if not accel_enum:
            # Try to match partial names
            for key, value in type_map.items():
                if key in accelerator_type:
                    accel_enum = value
                    break

        if not accel_enum:
            accel_enum = AcceleratorType.CPU if not fallback_to_generic else None
    else:
        accel_enum = accelerator_type

    # Create the appropriate monitor
    if accel_enum == AcceleratorType.NVIDIA:
        logger.info(f"Creating NVIDIA accelerator monitor for devices: {device_ids}")
        return NVIDIAAcceleratorMonitor(device_ids)

    elif accel_enum == AcceleratorType.CAMBRICON_MLU:
        logger.info(f"Creating MLU accelerator monitor for devices: {device_ids}")
        return MLUAcceleratorMonitor(device_ids)

    elif accel_enum == AcceleratorType.AMD:
        # AMD ROCm monitor - implementation would go here
        logger.info(f"Creating AMD accelerator monitor for devices: {device_ids}")
        # For now, use generic monitor for AMD
        return GenericAcceleratorMonitor(device_ids, "amd")

    elif accel_enum == AcceleratorType.MOCK:
        logger.info(f"Creating Mock accelerator monitor for devices: {device_ids}")
        return MockAcceleratorMonitor(device_ids)

    elif accel_enum == AcceleratorType.CPU:
        logger.info(f"Creating CPU monitor (no GPU) for devices: {device_ids}")
        return GenericAcceleratorMonitor(device_ids, "cpu")

    else:
        # Generic/unknown accelerator
        if fallback_to_generic:
            logger.info(
                f"Creating generic accelerator monitor for '{accelerator_type}', devices: {device_ids}"
            )
            return GenericAcceleratorMonitor(device_ids, str(accelerator_type))
        else:
            logger.warning(
                f"Unsupported accelerator type: {accelerator_type}, using CPU monitor"
            )
            return GenericAcceleratorMonitor(device_ids, "cpu")

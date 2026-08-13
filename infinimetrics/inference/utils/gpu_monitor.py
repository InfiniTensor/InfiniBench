#!/usr/bin/env python3
"""
GPU Monitoring Module - Revised Version
"""
import subprocess
import threading
import time
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class GPUMonitor(ABC):
    """Abstract base class for GPU monitoring"""

    def __init__(self, device_ids=None):
        """
        Initialize GPU monitor
        
        Args:
            device_ids: List of GPU device IDs to monitor. If None, monitor all GPUs.
        """
        self.device_ids = device_ids
        self.peak_memory_mib = 0  # Peak memory usage (MiB)
        self.monitor_thread = None
        self._stop_monitoring_flag = False  # Renamed variable for fix
        self.poll_interval = 0.5  # Polling interval (seconds)
        
        logger.info(f"GPUMonitor initialized for devices: {device_ids}")
    
    @abstractmethod
    def get_current_memory_usage(self):
        """Get current GPU memory usage for all devices"""
        pass
    
    def start_monitoring(self):
        """Start monitoring GPU memory usage"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        self._stop_monitoring_flag = False
        self.peak_memory_mib = 0
        
        def monitor_loop():
            logger.info("GPU monitoring started")
            while not self._stop_monitoring_flag:
                try:
                    current_mem = self.get_current_memory_usage()
                    if current_mem:
                        current_peak = max(current_mem)
                        if current_peak > self.peak_memory_mib:
                            self.peak_memory_mib = current_peak
                            logger.debug(f"New peak memory: {self.peak_memory_mib} MiB")
                except Exception as e:
                    logger.debug(f"Error getting GPU memory: {e}")
                
                time.sleep(self.poll_interval)
            
            logger.info("GPU monitoring stopped")
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring GPU memory usage"""
        self._stop_monitoring_flag = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                logger.warning("Monitor thread did not stop gracefully")
    
    def get_peak_memory_gb(self):
        """Get peak memory usage in GB"""
        return round(self.peak_memory_mib / 1024.0, 6)
    
    def get_peak_memory_mib(self):
        """Get peak memory usage in MiB"""
        return self.peak_memory_mib
    
    def get_peak_memory_per_device(self):
        """Get peak memory usage per device"""
        total_gb = self.get_peak_memory_gb()
        if self.device_ids:
            per_device = total_gb / len(self.device_ids)
            return {device_id: per_device for device_id in self.device_ids}
        else:
            return {"all": total_gb}

class NVIDIAGPUMonitor(GPUMonitor):
    """NVIDIA GPU monitor implementation"""
    
    def get_current_memory_usage(self):
        try:
            cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            
            if self.device_ids:
                device_str = ",".join(str(d) for d in self.device_ids)
                cmd.extend(["--id", device_str])
            
            out = subprocess.check_output(
                cmd,
                text=True, 
                stderr=subprocess.DEVNULL
            )
            
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            memory_values = [int(x) for x in lines if x.isdigit()]
            
            if memory_values:
                logger.debug(f"Current GPU memory usage: {memory_values} MiB")
                return memory_values
            else:
                return []
                
        except subprocess.CalledProcessError as e:
            logger.error(f"nvidia-smi command failed: {e}")
            return []
        except FileNotFoundError:
            logger.error("nvidia-smi not found. Is NVIDIA driver installed?")
            return []
        except Exception as e:
            logger.error(f"Error getting GPU memory: {e}")
            return []

class MockGPUMonitor(GPUMonitor):
    """Mock GPU monitor (for testing or CPU mode)"""
    
    def get_current_memory_usage(self):
        """Return mock memory usage data"""
        import random
        if self.device_ids:
            return [random.randint(100, 1000) for _ in self.device_ids]
        else:
            return [random.randint(100, 1000)]

def create_gpu_monitor(gpu_platform="nvidia", device_ids=None):
    """
    Factory function to create GPU monitor
    
    Args:
        gpu_platform: GPU platform name, "nvidia" or others
        device_ids: List of device IDs to monitor
        
    Returns:
        Instance of GPUMonitor
    """
    platform_lower = gpu_platform.lower()
    
    if platform_lower == "nvidia":
        logger.info(f"Creating NVIDIA GPU monitor for devices: {device_ids}")
        return NVIDIAGPUMonitor(device_ids)
    elif platform_lower == "mock" or platform_lower == "test":
        logger.info(f"Creating Mock GPU monitor for devices: {device_ids}")
        return MockGPUMonitor(device_ids)
    else:
        logger.warning(f"Unsupported GPU platform: {gpu_platform}, using mock monitor")
        return MockGPUMonitor(device_ids)

import subprocess
import threading
import time
from abc import ABC, abstractmethod

class GPUMonitor(ABC):
    """Abstract base class for GPU monitoring"""
    
    def __init__(self):
        self.peak_memory = 0
    
    @abstractmethod
    def get_current_memory_usage(self):
        """Get current GPU memory usage"""
        pass
    
    def monitor_process(self, process, poll_interval=0.5):
        """Monitor GPU memory usage of a process"""
        def monitor_loop():
            while True:
                if process.poll() is not None:
                    break
                try:
                    current_mem = self.get_current_memory_usage()
                    if current_mem:
                        self.peak_memory = max(self.peak_memory, max(current_mem))
                except Exception:
                    pass
                time.sleep(poll_interval)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        return monitor_thread
    
    def get_peak_memory_gb(self):
        """Get peak memory usage in GB"""
        return round(self.peak_memory / 1024.0, 6)
    
    def get_peak_memory_mib(self):
        """Get peak memory usage in MiB"""
        return self.peak_memory


class NVIDIAGPUMonitor(GPUMonitor):
    """NVIDIA GPU monitor implementation"""
    
    def get_current_memory_usage(self):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL
            )
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            return [int(x) for x in lines] if lines else []
        except Exception:
            return []


class OtherPlatformGPUMonitor(GPUMonitor):
    """Other platform GPU monitor implementation (example)"""
    
    def get_current_memory_usage(self):
        """GPU memory monitoring implementation for other platforms"""
        # Can be extended to support other platforms like NPU MLU, etc.
        # Currently returns empty list, need implementation for specific platform
        return []


def create_gpu_monitor(platform="nvidia"):
    """Factory function to create GPU monitor"""
    if platform.lower() == "nvidia":
        return NVIDIAGPUMonitor()
    else:
        return OtherPlatformGPUMonitor()

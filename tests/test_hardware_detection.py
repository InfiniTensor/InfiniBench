from types import SimpleNamespace

from infinibench.common import hardware_info
from infinibench.common.hardware_info import HardwareCollector
from infinibench.utils import hardware_detector
from infinibench.utils.hardware_detector import HardwareDetector


MTHREADS_OUTPUT = """
Attached GPUs                                     :  2

GPU0 00000000:03:00.0
    Product Name                                  :  MTT S5000
    Product Brand                                 :  MTT
    GPU UUID                                      :  first-uuid
    GPU Link Info

GPU1 00000000:05:00.0
    Product Name                                  :  MTT S5000
    Product Brand                                 :  MTT
    GPU UUID                                      :  second-uuid
    GPU Link Info
"""


def _empty_hardware():
    return {
        "gpu_count": 0,
        "gpu_model": "Unknown",
        "cuda_version": "Unknown",
    }


def _successful_probe(*args, **kwargs):
    return SimpleNamespace(returncode=0, stdout=MTHREADS_OUTPUT)


def test_hardware_collector_counts_only_mthreads_device_headers(monkeypatch):
    monkeypatch.setattr(hardware_info, "_which", lambda command: command)
    monkeypatch.setattr(hardware_info.subprocess, "run", _successful_probe)
    monkeypatch.setattr(HardwareCollector, "_collect_musa_version", lambda self: None)
    hardware = _empty_hardware()

    result = HardwareCollector()._probe_mthreads("moore", hardware)

    assert result.success
    assert result.count == 2
    assert hardware["gpu_count"] == 2
    assert hardware["gpu_model"] == "MTT S5000"


def test_hardware_detector_counts_only_mthreads_device_headers(monkeypatch):
    monkeypatch.setattr(hardware_detector, "_which", lambda command: command)
    monkeypatch.setattr(hardware_detector.subprocess, "run", _successful_probe)
    hardware = _empty_hardware()

    assert HardwareDetector._probe_mthreads(hardware)
    assert hardware["gpu_count"] == 2
    assert hardware["gpu_model"] == "MTT S5000"

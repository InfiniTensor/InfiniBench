"""
Framework-specific training runners
"""

from .megatron_runner import MegatronRunner

# InfinitrainRunner is optional, import only if available
try:
    from .infinitrain_runner import InfinitrainRunner
    __all__ = ['MegatronRunner', 'InfinitrainRunner']
except ImportError:
    __all__ = ['MegatronRunner']

"""Framework-specific training implementations."""

from .megatron_impl import MegatronImpl

# InfinitrainImpl is optional, import only if available
try:
    from .infinitrain_impl import InfinitrainImpl

    __all__ = ["MegatronImpl", "InfinitrainImpl"]
except ImportError:
    __all__ = ["MegatronImpl"]

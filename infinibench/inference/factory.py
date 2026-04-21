#!/usr/bin/env python3
"""
Factory helpers for inference framework components
"""


def create_framework_impl(config):
    """
    Create model adapter for direct inference.
    """
    if config.framework == "infinilm":
        from .frameworks.infinilm_impl import InfiniLMImpl

        return InfiniLMImpl(config)
    else:
        from .frameworks.vllm_impl import VLLMImpl

        return VLLMImpl(config)


def create_service_manager(config):
    """
    Create service manager for service inference.
    """
    if config.framework == "infinilm":
        from .service_manager import InfiniLMServiceManager

        return InfiniLMServiceManager(config)
    else:
        from .service_manager import VLLMServiceManager

        return VLLMServiceManager(config)

#!/usr/bin/env python3
"""
Inference Runner Factory Class
Creates appropriate Runner and Adapter based on configuration
"""

import logging
import sys
import os
from typing import Dict, Tuple

from infer_config import InferConfig, InferMode, FrameworkType
from infer_runner_base import InferRunnerBase
from adapter_base import InferAdapter

logger = logging.getLogger(__name__)

class InferRunnerFactory:
    """Inference Runner Factory"""
    
    @staticmethod
    def create_runner_and_adapter(config: InferConfig) -> Tuple[InferRunnerBase, InferAdapter]:
        """
        Create Runner and Adapter
        
        Returns: (runner, adapter)
        """
        # Create adapter based on framework
        adapter = InferRunnerFactory.create_adapter(config)
        
        # Create runner based on mode
        runner = InferRunnerFactory.create_runner(config, adapter)
        
        return runner, adapter
    
    @staticmethod
    def create_adapter(config: InferConfig) -> InferAdapter:
        """Create adapter"""
        logger.info(f"Creating adapter for framework: {config.framework.value}")
        
        if config.framework == FrameworkType.INFINILM:
            try:
                from adapters.infinilm_adapter import InfiniLMAdapter
                adapter = InfiniLMAdapter(config)
                logger.info("InfiniLMAdapter created successfully")
            except ImportError as e:
                logger.error(f"Failed to import InfiniLMAdapter: {e}")
                raise ImportError("InfiniLMAdapter is not available. Please check if InfiniLM is installed.")
        
        elif config.framework == FrameworkType.VLLM:
            try:
                from adapters.vllm_adapter import VLLMAdapter
                adapter = VLLMAdapter(config)
                logger.info("VLLMAdapter created successfully")
            except ImportError as e:
                logger.error(f"Failed to import VLLMAdapter: {e}")
                raise ImportError("VLLMAdapter is not available. Please check if vLLM is installed.")
        
        else:
            raise ValueError(f"Unsupported framework: {config.framework}")
        
        # Validate adapter configuration
        errors = adapter.validate_config()
        if errors:
            error_msg = "Adapter configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.warning(error_msg)
        
        return adapter
    
    @staticmethod
    def create_runner(config: InferConfig, adapter: InferAdapter) -> InferRunnerBase:
        """Create runner"""
        logger.info(f"Creating runner for mode: {config.mode.value}")
        
        if config.mode == InferMode.DIRECT:
            try:
                from direct_infer_runner import DirectInferRunner
                runner = DirectInferRunner(config, adapter)
                logger.info("DirectInferRunner created successfully")
            except ImportError as e:
                logger.error(f"Failed to import DirectInferRunner: {e}")
                raise
        
        elif config.mode == InferMode.SERVICE:
            try:
                from service_infer_runner import ServiceInferRunner
                runner = ServiceInferRunner(config, adapter)
                logger.info("ServiceInferRunner created successfully")
            except ImportError as e:
                logger.error(f"Failed to import ServiceInferRunner: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported inference mode: {config.mode}")
        
        return runner
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check dependencies"""
        dependencies = {
            "infinilm": False,
            "vllm": False,
            "numpy": False,
            "torch": False
        }
        
        # Check InfiniLM
        try:
            # Try importing InfiniLM related modules
            import sys
            import os
            
            # Check if in InfiniLM directory
            if os.path.exists("scripts/jiuge.py"):
                dependencies["infinilm"] = True
            else:
                # Try to determine via environment variable
                infinilm_path = os.environ.get("INFINILM_PATH", "")
                if infinilm_path and os.path.exists(os.path.join(infinilm_path, "scripts/jiuge.py")):
                    dependencies["infinilm"] = True
        except:
            pass
        
        # Check vLLM
        try:
            import vllm
            dependencies["vllm"] = True
        except ImportError:
            pass
        
        # Check numpy
        try:
            import numpy
            dependencies["numpy"] = True
        except ImportError:
            pass
        
        # Check torch
        try:
            import torch
            dependencies["torch"] = True
        except ImportError:
            pass
        
        return dependencies

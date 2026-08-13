#!/usr/bin/env python3
"""
Inference Configuration Manager
Parses config.json, identifies direct/service mode, identifies infinilm/vllm framework
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import random
import string

logger = logging.getLogger(__name__)

class InferMode(Enum):
    """Inference mode enumeration"""
    DIRECT = "direct"
    SERVICE = "service"

class FrameworkType(Enum):
    """Framework type enumeration"""
    INFINILM = "infinilm"
    VLLM = "vllm"

@dataclass
class ParallelConfig:
    """Parallel configuration"""
    dp: int = 1
    tp: int = 1
    pp: int = 1
    sp: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParallelConfig':
        """Create parallel configuration from dictionary"""
        if not data:
            return cls()
        return cls(
            dp=data.get("dp", 1),
            tp=data.get("tp", 1),
            pp=data.get("pp", 1),
            sp=data.get("sp", 1)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dp": self.dp,
            "tp": self.tp,
            "pp": self.pp,
            "sp": self.sp
        }

@dataclass
class DeviceConfig:
    """Device configuration"""
    gpu_platform: str = "nvidia"
    device_ids: List[int] = None
    cpu_only: bool = False

    def __post_init__(self):
        if self.device_ids is None:
            self.device_ids = [0]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceConfig':
        """Create device configuration from dictionary"""
        if not data:
            return cls()
        return cls(
            gpu_platform=data.get("gpu_platform", "nvidia"),
            device_ids=data.get("device_ids", [0]),
            cpu_only=data.get("cpu_only", False)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "gpu_platform": self.gpu_platform,
            "device_ids": self.device_ids,
            "cpu_only": self.cpu_only
        }

@dataclass
class DirectInferArgs:
    """Direct inference arguments"""
    parallel: ParallelConfig
    static_batch_size: int
    prompt_token_num: int
    output_token_num: int = 128
    max_seq_len: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DirectInferArgs':
        """Create direct inference arguments from dictionary"""
        return cls(
            parallel=ParallelConfig.from_dict(data.get("parallel", {})),
            static_batch_size=data.get("static_batch_size", 1),
            prompt_token_num=data.get("prompt_token_num", 1024),
            output_token_num=data.get("output_token_num", 128),
            max_seq_len=data.get("max_seq_len", 4096),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 50)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "parallel": self.parallel.to_dict(),
            "static_batch_size": self.static_batch_size,
            "prompt_token_num": self.prompt_token_num,
            "output_token_num": self.output_token_num,
            "max_seq_len": self.max_seq_len,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

@dataclass
class ServiceInferArgs:
    """Service inference arguments"""
    parallel: ParallelConfig
    request_trace: str
    concurrency: int = 32
    max_seq_len: int = 4096
    stream: bool = True
    timeout_ms: int = 30000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInferArgs':
        """Create service inference arguments from dictionary"""
        return cls(
            parallel=ParallelConfig.from_dict(data.get("parallel", {})),
            request_trace=data.get("request_trace", ""),
            concurrency=data.get("concurrency", 32),
            max_seq_len=data.get("max_seq_len", 4096),
            stream=data.get("stream", True),
            timeout_ms=data.get("timeout_ms", 30000)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "parallel": self.parallel.to_dict(),
            "request_trace": self.request_trace,
            "concurrency": self.concurrency,
            "max_seq_len": self.max_seq_len,
            "stream": self.stream,
            "timeout_ms": self.timeout_ms
        }

@dataclass
class InferConfig:
    """Main inference configuration class"""
    # Basic information
    run_id: str
    testcase: str
    model: str
    model_path: str
    model_config: Optional[str]

    # Dataset
    train_dataset: Optional[str]
    validation_dataset: Optional[str]
    test_dataset: Optional[str]

    # Output
    output_dir: str

    # Execution mode
    mode: InferMode
    framework: FrameworkType

    # Device configuration
    device: DeviceConfig

    # Inference arguments (varies by mode)
    infer_args: Any  # DirectInferArgs or ServiceInferArgs

    # Execution parameters
    timeout_ms: int
    warmup_iterations: int
    measured_iterations: int

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'InferConfig':
        """Create configuration object from dictionary - implemented with clear logic"""
        # 1. Read outer level
        outer_run_id = config_dict.get("run_id")
        outer_testcase = config_dict.get("testcase")
        config_data = config_dict.get("config", {})
        
        # 2. Read inner level (for error checking)
        inner_run_id = config_data.get("run_id")
        inner_testcase = config_data.get("testcase")
        
        # 3. Process testcase
        # Rule 2.1: If outer level has testcase → use it
        if outer_testcase:
            testcase = outer_testcase
            logger.info(f"Using outer testcase: {testcase}")

        # Rule 2.2: If inner level has testcase → raise error
        elif inner_testcase:
            raise ValueError(
                "testcase must be at the outer level, not inside 'config'. "
                f"Found: '{inner_testcase}' inside 'config'. "
                "Please move it to the outer level."
            )
        
        # Rule 2.3: If no testcase at either level → raise error
        else:
            raise ValueError(
                "testcase is required at the outer level of the config. "
                "Example: {\"testcase\": \"infer.InfiniLM.Direct\", ...}"
            )
        
        testcase = outer_testcase
        logger.info(f"Using testcase: {testcase}")
        
        # 4. Process run_id
        # Rule 1.2: If inner level has run_id → raise error
        if inner_run_id:
            raise ValueError(
                "run_id must be at the outer level, not inside 'config'. "
                f"Found: '{inner_run_id}' inside 'config'. "
                "Please move it to the outer level or remove it to auto-generate."
            )
        
        # Rule 1.1: If outer level has run_id → use it (add timestamp+random code to prevent overwriting)
        elif outer_run_id:
            run_id = cls._enhance_user_run_id(outer_run_id)
            logger.info(f"Using enhanced user-provided run_id: {run_id}")
        else:
            # Rule 1.3: Auto-generate run_id
            run_id = cls._generate_auto_run_id(testcase)
            logger.info(f"Auto-generated run_id: {run_id}")
        
        # 5. Parse mode and framework from testcase
        testcase_lower = testcase.lower()
        
        # Determine inference mode
        if "service" in testcase_lower:
            mode = InferMode.SERVICE
        elif "direct" in testcase_lower:
            mode = InferMode.DIRECT
        else:
            mode = InferMode.DIRECT
        
        # Determine framework
        if "vllm" in testcase_lower:
            framework = FrameworkType.VLLM
        elif "infinilm" in testcase_lower:
            framework = FrameworkType.INFINILM
        else:
            raise ValueError(f"Cannot determine framework from testcase: {testcase}")
        
        # 6. Parse model_path
        model_path = config_data.get("model_path")
        model_config = config_data.get("model_config")
        
        if not model_path and model_config:
            model_path = str(Path(model_config).parent)
            logger.info(f"Inferred model_path from model_config: {model_path}")
        elif not model_path:
            raise ValueError("Either model_path or model_config must be provided in config")
        
        # 7. Parse inference arguments
        infer_args_dict = config_data.get("infer_args", {})
        if mode == InferMode.DIRECT:
            infer_args = DirectInferArgs.from_dict(infer_args_dict)
        else:
            infer_args = ServiceInferArgs.from_dict(infer_args_dict)
        
        # 8. Parse device configuration
        device_config = DeviceConfig.from_dict(config_data.get("device", {}))
        
        return cls(
            run_id=run_id,
            testcase=testcase,
            model=config_data.get("model", "unknown"),
            model_path=model_path,
            model_config=model_config,
            
            train_dataset=config_data.get("train_dataset"),
            validation_dataset=config_data.get("validation_dataset"),
            test_dataset=config_data.get("test_dataset"),
            
            output_dir=config_data.get("output_dir", "./test_output"),
            
            mode=mode,
            framework=framework,
            device=device_config,
            infer_args=infer_args,
            
            timeout_ms=config_data.get("timeout_ms", 30000)if mode == InferMode.SERVICE else None,
            warmup_iterations=config_data.get("warmup_iterations", 10),
            measured_iterations=config_data.get("measured_iterations", 100)
        )
    
    def _enhance_user_run_id(user_run_id: str) -> str:
        """
        Enhance user-provided run_id by adding timestamp and random code to prevent overwriting

        Args:
            user_run_id: User-provided run_id

        Returns:
            Enhanced run_id: {user_run_id}.{timestamp}.{random8}
        """
        # If already contains timestamp and random code, return directly (prevent duplicate addition)
        import re
        timestamp_pattern = r'\.\d{8}_\d{6}\.[a-z0-9]{8}$'
        if re.search(timestamp_pattern, user_run_id):
            logger.info(f"User run_id already contains timestamp and random code: {user_run_id}")
            return user_run_id

        # Add timestamp and random code
        from datetime import datetime
        import random
        import string

        # Clean user run_id
        cleaned_user_id = user_run_id.strip().strip(".").replace("..", ".")

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 8-character random code
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

        # Combine
        enhanced_run_id = f"{cleaned_user_id}.{timestamp}.{random_suffix}"

        logger.info(f"Enhanced user run_id: {user_run_id} -> {enhanced_run_id}")
        return enhanced_run_id

    @staticmethod
    def _generate_auto_run_id(testcase: str) -> str:
        """
        Auto-generate run_id
        Format: {testcase}.{timestamp}.{random8}
        Example: infer.InfiniLM.Direct.20251210_143025.a1b2c3d4
        """
        # Clean testcase
        cleaned_testcase = testcase.strip().strip(".").replace("..", ".")
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 8-character random code
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Combine
        run_id = f"{cleaned_testcase}.{timestamp}.{random_suffix}"
        
        return run_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON output)"""
        return {
            "run_id": self.run_id,
            "testcase": self.testcase,
            "mode": self.mode.value,
            "framework": self.framework.value,
            "model": self.model,
            "model_path": self.model_path,
            "model_config": self.model_config,
            "device": self.device.to_dict(),
            "infer_args": self.infer_args.to_dict() if hasattr(self.infer_args, 'to_dict') else {},
            "warmup_iterations": self.warmup_iterations,
            "measured_iterations": self.measured_iterations
        }
        if self.mode == InferMode.SERVICE and self.timeout_ms is not None:
            result["timeout_ms"] = self.timeout_ms
        
        return result

class InferConfigManager:
    """Inference Configuration Manager"""
    
    @staticmethod
    def load_config(config_file: str) -> InferConfig:
        """Load configuration from config file"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Support single config or config list
            if isinstance(config_data, dict):
                config_dict = config_data
            elif isinstance(config_data, list) and len(config_data) > 0:
                config_dict = config_data[0]
            else:
                raise ValueError(f"Invalid config format in {config_file}")
            
            # Create configuration object
            config = InferConfig.from_dict(config_dict)
            
            # Validate configuration
            errors = InferConfigManager.validate_config(config)
            if errors:
                error_msg = "Configuration validation errors:\n" + "\n".join(f"  - {e}" for e in errors)
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON config file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            raise
    
    @staticmethod
    def validate_config(config: InferConfig) -> List[str]:
        """Validate configuration effectiveness"""
        errors = []
        
        # Basic validation
        if not config.run_id:
            errors.append("run_id cannot be empty")
        
        if not config.testcase:
            errors.append("testcase cannot be empty")
        
        if not config.model or config.model == "unknown":
            errors.append("model name is required")
        
        # Model path validation
        model_dir = Path(config.model_path)
        if not model_dir.exists():
            errors.append(f"Model directory does not exist: {model_dir}")
        else:
            # Check config.json
            config_file = model_dir / "config.json"
            if not config_file.exists():
                errors.append(f"config.json not found in model directory: {model_dir}")
        
        # Mode-specific validation
        if config.mode == InferMode.DIRECT:
            if not isinstance(config.infer_args, DirectInferArgs):
                errors.append("Direct mode requires DirectInferArgs")
            else:
                if config.infer_args.static_batch_size <= 0:
                    errors.append("Batch size must be positive")
                if config.infer_args.prompt_token_num <= 0:
                    errors.append("Prompt token number must be positive")
        
        elif config.mode == InferMode.SERVICE:
            if not isinstance(config.infer_args, ServiceInferArgs):
                errors.append("Service mode requires ServiceInferArgs")
            else:
                if not config.infer_args.request_trace:
                    errors.append("Request trace is required for service mode")
                if config.infer_args.concurrency <= 0:
                    errors.append("Concurrency must be positive")
                
                # Verify trace file exists
                trace_path = Path(config.infer_args.request_trace)
                if not trace_path.exists():
                    errors.append(f"Trace file not found: {trace_path}")
        
        # Output directory validation
        output_dir = Path(config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Output directory is not writable: {e}")
        
        return errors

    @staticmethod
    def generate_auto_run_id(testcase: str) -> str:
        """
        Auto-generate run_id (public method)
        
        Format: {testcase}.{timestamp}.{random8}
        Example: infer.InfiniLM.Direct.20251210_143025.a1b2c3d4
        
        Args:
            testcase: testcase string
            
        Returns:
            Generated run_id
        """
        # Clean testcase
        cleaned_testcase = testcase.strip().strip(".").replace("..", ".")
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 8-character random code
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Combine
        run_id = f"{cleaned_testcase}.{timestamp}.{random_suffix}"
        
        return run_id
    
    # Original private method calls public method
    @staticmethod
    def _generate_auto_run_id(testcase: str) -> str:
        """Private method, calls public method (maintain backward compatibility)"""
        return InferConfig.generate_auto_run_id(testcase)
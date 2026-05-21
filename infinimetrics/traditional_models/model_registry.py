#!/usr/bin/env python3
"""Model registry for traditional model benchmarks.

Enumerates all models from the PyTorchModels/ directory with their
category, name, execution scripts, and data requirements.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Reference to PyTorchModels root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PYTORCH_MODELS_DIR = _REPO_ROOT / "PyTorchModels"

# All models organized by category
MODEL_REGISTRY: List[Dict[str, Any]] = [
    # Detection
    {
        "category": "Detection",
        "name": "fasterrcnn",
        "path": "Detection/fasterrcnn",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Detection/data/VOCdevkit",
        "env_vars": {"DATA_DIR": "../data/VOCdevkit"},
    },
    {
        "category": "Detection",
        "name": "ssd",
        "path": "Detection/ssd",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Detection/data/VOCdevkit",
        "env_vars": {"DATA_DIR": "../data/VOCdevkit"},
    },
    {
        "category": "Detection",
        "name": "yolo",
        "path": "Detection/yolo",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Detection/data/coco",
        "env_vars": {"MODEL": "yolov5s", "DATA_DIR": "../data/coco"},
    },
    # ImageClassification
    {
        "category": "ImageClassification",
        "name": "torchvision",
        "path": "ImageClassification/TorchVision",
        "train_script": "run_all_models_train.sh",
        "eval_script": None,
        "data_dir": "ImageClassification/data/imagenet2012",
        "env_vars": {"DATA_DIR": "../data/imagenet2012"},
    },
    # GAN
    {
        "category": "GAN",
        "name": "dcgan",
        "path": "GAN/dcgan",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "GAN/data/lsun",
        "env_vars": {"DATA_DIR": "../data/lsun"},
    },
    # NLP
    {
        "category": "NLP",
        "name": "huggingface",
        "path": "NLP/HuggingFace",
        "train_script": "run_train_online.sh",
        "eval_script": None,
        "data_dir": None,
        "env_vars": {},
    },
    # RL
    {
        "category": "RL",
        "name": "dqn",
        "path": "RL/dqn",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": None,
        "env_args": ["checkpoints", "100", "0.0001"],
        "env_vars": {},
    },
    # Recommendation
    {
        "category": "Recommendation",
        "name": "dlrm",
        "path": "Recommendation/DLRM",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Recommendation/data/ml-20mx4x16",
        "env_vars": {"DATA_DIR": "../data/ml-20mx4x16"},
    },
    # Super Resolution
    {
        "category": "SR",
        "name": "espcn",
        "path": "SR/ESPCN",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "SR/ESPCN/data",
        "env_vars": {},
    },
    # Segmentation
    {
        "category": "Segmentation",
        "name": "deeplab",
        "path": "Segmentation/deeplab",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Segmentation/data",
        "env_vars": {},
    },
    {
        "category": "Segmentation",
        "name": "fcn",
        "path": "Segmentation/fcn",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Segmentation/data",
        "env_vars": {},
    },
    {
        "category": "Segmentation",
        "name": "lraspp",
        "path": "Segmentation/lraspp",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Segmentation/data",
        "env_vars": {},
    },
    {
        "category": "Segmentation",
        "name": "unet",
        "path": "Segmentation/unet",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Segmentation/data",
        "env_vars": {},
    },
    # Speech
    {
        "category": "Speech",
        "name": "deepspeech2",
        "path": "Speech/deepspeech2",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "Speech/data",
        "env_vars": {},
    },
    {
        "category": "Speech",
        "name": "wav2vec",
        "path": "Speech/wav2vec",
        "train_script": "run_train_online.sh",
        "eval_script": None,
        "data_dir": None,
        "env_vars": {},
    },
    # TimeSeriesPrediction
    {
        "category": "TimeSeriesPrediction",
        "name": "lstm",
        "path": "TimeSeriesPrediction/lstm",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "TimeSeriesPrediction/data",
        "env_args": ["../data/complete_data.csv", "200", "512", "0.0001"],
        "env_vars": {},
    },
    {
        "category": "TimeSeriesPrediction",
        "name": "tcn",
        "path": "TimeSeriesPrediction/tcn",
        "train_script": "run_train_val.sh",
        "eval_script": None,
        "data_dir": "TimeSeriesPrediction/data",
        "env_vars": {},
    },
    # InstanceSeg
    {
        "category": "InstanceSeg",
        "name": "maskrcnn",
        "path": "InstanceSeg/maskrcnn",
        "train_script": "run_train.sh",
        "eval_script": None,
        "data_dir": "InstanceSeg/data",
        "env_vars": {},
    },
]

# Supported platforms for env.sh
PLATFORM_MAP = {
    "nvidia": "NVIDIA_GPU",
    "cuda": "NVIDIA_GPU",
    "metax": "METAX_GPU",
    "metax_gpu": "METAX_GPU",
    "cambricon": "CAMBRICON_MLU",
    "cambricon_mlu": "CAMBRICON_MLU",
    "ascend": "ASCEND_NPU",
    "ascend_npu": "ASCEND_NPU",
    "moore": "MOORE_GPU",
    "moore_gpu": "MOORE_GPU",
    "hygon": "SUGON_DCU",
    "sugon_dcu": "SUGON_DCU",
    "iluvatar": "ILLUVATAR_GPU",
    "iluvatar_gpu": "ILLUVATAR_GPU",
}


def get_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Get model info by name."""
    for model in MODEL_REGISTRY:
        if model["name"] == model_name:
            return model.copy()
    return None


def get_models_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all models in a category."""
    return [m.copy() for m in MODEL_REGISTRY if m["category"] == category]


def get_all_categories() -> List[str]:
    """Get all unique categories."""
    seen = set()
    result = []
    for m in MODEL_REGISTRY:
        if m["category"] not in seen:
            seen.add(m["category"])
            result.append(m["category"])
    return result


def get_all_models() -> List[Dict[str, Any]]:
    """Get all registered models."""
    return [m.copy() for m in MODEL_REGISTRY]


def get_platform_env_name(platform: str) -> str:
    """Get the env.sh PLATFORM_ENV value for a platform."""
    return PLATFORM_MAP.get(platform.lower(), "NVIDIA_GPU")

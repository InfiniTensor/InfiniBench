# InfiniBench  
**Universal End-to-End Model and Operator Test Framework**  

[![PyPI Version](https://img.shields.io/pypi/v/infiniMOT)](https://pypi.org/project/infiniMOT/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features  
- 🧩 **Extensible Architecture**: Support multiple backends (PyTorch, Triton, CUDA)  
- ✅ **Model Testing**: Validate model implementations (e.g., `src/models/Llama`)  
- 🛠️ **Operator Testing**: Cross-backend operator verification  
- 🔧 **Toolkit**: Utilities for inference, profiling, and result analysis  

## Quickstart  
```bash
# Install with PyTorch backend
pip install -e ".[torch]"  

# Run a test  
python -m infiniBench run-test --model GPT2 --backend torch

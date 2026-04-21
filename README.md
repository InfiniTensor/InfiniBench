<div align="center">

# InfiniBench

**An InfiniTensor-Featured Comprehensive Accelerator Evaluation Framework**

[![Format Check](https://img.shields.io/badge/Format_Check-passing-success)](https://github.com/InfiniTensor/InfiniBench)
[![Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniBench)](https://github.com/InfiniTensor/InfiniBench/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniBench)](https://github.com/InfiniTensor/InfiniBench/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/InfiniTensor/InfiniBench/blob/master/LICENSE)

A unified, modular testing framework for comprehensive performance evaluation of accelerator hardware and software stacks.

</div>

---

## 🎯 Overview

**InfiniBench** provides standardized interfaces for benchmarking across multiple layers:

- **Hardware-Level**: GPU memory bandwidth, cache performance, compute capabilities
- **Operator-Level**: Individual operation performance (FLOPS, latency)
- **Inference-Level**: End-to-end model inference throughput and latency
- **Communication-Level**: NCCL collective operations and inter-GPU communication

### Key Features

- **Unified Adapter Interface** - Consistent API across all test types and frameworks
- **Extensible Architecture** - Easy to add new test types, frameworks, and metrics
- **Comprehensive Metrics** - Scalar values, time-series data, custom measurements
- **Framework Agnostic** - Support for InfiniLM, vLLM, InfiniCore, and more
- **Production Ready** - Robust error handling, logging, and result aggregation

---

## 📚 Documentation

For detailed guides, configuration, and examples, see the [full documentation](./docs).

### Quick Links

- [Installation Guide](./docs/installation.md) - Prerequisites and dependencies
- [Configuration](./docs/configuration.md) - Input format and parameters
- [Development Guide](./docs/development.md) - Development setup and extending the framework

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](./docs/development.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

<div align="center">

**Built with ❤️ by the InfiniTensor Team**

[Website](https://infinitensor.org) | [Documentation](./docs) | [GitHub](https://github.com/InfiniTensor)

</div>

# Development Guide

This guide explains how to set up the development environment and extend InfiniBench by adding new adapters and metrics.

## Development Setup

### Pre-commit Setup

This project uses **pre-commit** to automatically enforce code style (Black) and linting (Flake8) before every commit.

#### Installation

```bash
pip install pre-commit
```

#### Setup (Run Once)

Run the following command in the project root directory to activate the git hooks:

```bash
pre-commit install
```

*Output should be: `pre-commit installed at .git/hooks/pre-commit`*

#### Usage

- **Automatic:** Just run `git commit` as usual.
  - If **Black** modifies your files: Run `git add .` again and re-commit.
  - If **Flake8** reports errors: Fix the errors, `git add`, and re-commit.
- **Manual:** To check all files in the repository without committing:
  ```bash
  pre-commit run --all-files
  ```

#### Skip Checks (Emergency Only)

If you need to bypass the checks for a specific commit:

```bash
git commit -m "your message" --no-verify
```

---

## Adding a New Adapter

### 1. Create Adapter Class

Create a new adapter class by inheriting from `BaseAdapter`:

```python
from infinibench.adapter import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your adapter
        self.device = config.get('device', 'nvidia')

    def setup(self):
        # Prepare test environment
        print(f"Setting up {self.__class__.__name__}")
        # Load models, allocate memory, etc.

    def process(self, test_input):
        # Execute test and return metrics
        results = {
            "my.metric": {
                "value": 42.0,
                "unit": "operations/s"
            }
        }
        return results

    def teardown(self):
        # Cleanup resources
        print(f"Tearing down {self.__class__.__name__}")
        # Free memory, close connections, etc.
```

### 2. Register Adapter in Dispatcher

Add your adapter to the adapter registry in `dispatcher.py`:

```python
# In dispatcher.py
self.adapter_registry = {
    ("operator", "myframework"): MyCustomAdapter,
    # ... existing adapters ...
}
```

### 3. Define Test Case and Metrics

Create a JSON configuration file:

```json
{
    "run_id": "my_test",
    "testcase": "operator.myframework.MyTest",
    "config": {
        "device": "nvidia",
        "iterations": 100
    },
    "metrics": [
        {"name": "my.metric"}
    ]
}
```

### 4. Test Your Adapter

```bash
python main.py my_test_config.json
```

## Adding New Metrics

### Define Metric Class

In `infinibench/common/metrics.py`:

```python
class CustomMetric(Metric):
    def __init__(self, name: str, value: float, unit: str = ""):
        super().__init__(name, value, unit)

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp
        }
```

### Use Custom Metric

In your adapter's `process` method:

```python
def process(self, test_input):
    metric = CustomMetric("custom.metric", 123.45, "ms")
    return {"custom.metric": metric.to_dict()}
```

## Adapter Interface Reference

### BaseAdapter Methods

| Method | Description | Required |
|--------|-------------|----------|
| `__init__(config)` | Initialize adapter with configuration | Yes |
| `setup()` | Prepare test environment | Yes |
| `process(test_input)` | Execute test and return metrics | Yes |
| `teardown()` | Cleanup resources | Yes |

### Test Input Structure

```python
{
    "run_id": "unique_identifier",
    "testcase": "category.framework.test_name",
    "config": {...},
    "metrics": [...]
}
```

### Metric Return Format

```python
{
    "metric.name": {
        "value": 42.0,
        "unit": "unit_name"
    }
}
```

## Code Organization

### Directory Structure

```
infinibench/
├── hardware/       # Hardware test adapters
├── operators/      # Operator test adapters
├── inference/      # Inference test adapters
├── communication/           # Communication test adapters
└── common/         # Shared utilities
```

### Naming Conventions

- **Adapter files**: `{framework}_adapter.py`
- **Adapter classes**: `{Framework}Adapter` (e.g., `InfiniCoreAdapter`)
- **Test cases**: `<category>.<framework>.<TestName>`

## Best Practices

1. **Error Handling**: Always wrap critical operations in try-except blocks
2. **Logging**: Use Python's logging module for debug output
3. **Resource Management**: Ensure `teardown()` properly releases all resources
4. **Configuration**: Provide sensible defaults for all config parameters
5. **Documentation**: Add docstrings to all public methods

## Testing Your Changes

1. Create a test configuration file
2. Run with `--verbose` flag for detailed output
3. Check output directory for metrics.json
4. Verify logs for any errors

```bash
python main.py test_config.json --verbose
```

## Contributing

When contributing adapters or metrics:

1. Follow existing code style
2. Add documentation for new features
3. Include example configurations
4. Update relevant documentation files

For questions or discussions, please open an issue on GitHub.

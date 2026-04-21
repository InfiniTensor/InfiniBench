# 开发指南

本指南解释如何通过添加新适配器和指标来扩展 InfiniBench。

## 添加新适配器

### 1. 创建适配器类

通过继承 `BaseAdapter` 创建新适配器类：

```python
from infinibench.adapter import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    def __init__(self, config):
        super().__init__(config)
        # 初始化适配器
        self.device = config.get('device', 'nvidia')

    def setup(self):
        # 准备测试环境
        print(f"正在设置 {self.__class__.__name__}")
        # 加载模型、分配内存等

    def process(self, test_input):
        # 执行测试并返回指标
        results = {
            "my.metric": {
                "value": 42.0,
                "unit": "operations/s"
            }
        }
        return results

    def teardown(self):
        # 清理资源
        print(f"正在清理 {self.__class__.__name__}")
        # 释放内存、关闭连接等
```

### 2. 在调度器中注册适配器

在 `dispatcher.py` 中将适配器添加到适配器注册表：

```python
# 在 dispatcher.py 中
self.adapter_registry = {
    ("operator", "myframework"): MyCustomAdapter,
    # ... 现有适配器 ...
}
```

### 3. 定义测试用例和指标

创建 JSON 配置文件：

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

### 4. 测试适配器

```bash
python main.py my_test_config.json
```

## 添加新指标

### 定义指标类

在 `infinibench/common/metrics.py` 中：

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

### 使用自定义指标

在适配器的 `process` 方法中：

```python
def process(self, test_input):
    metric = CustomMetric("custom.metric", 123.45, "ms")
    return {"custom.metric": metric.to_dict()}
```

## 适配器接口参考

### BaseAdapter 方法

| 方法 | 描述 | 必需 |
|--------|-------------|----------|
| `__init__(config)` | 使用配置初始化适配器 | 是 |
| `setup()` | 准备测试环境 | 是 |
| `process(test_input)` | 执行测试并返回指标 | 是 |
| `teardown()` | 清理资源 | 是 |

### 测试输入结构

```python
{
    "run_id": "unique_identifier",
    "testcase": "category.framework.test_name",
    "config": {...},
    "metrics": [...]
}
```

### 指标返回格式

```python
{
    "name": "metric.name",
    "value": 42.0,
    "unit": "unit_name"
}
```

## 代码组织

### 目录结构

```
infinibench/
├── hardware/       # 硬件测试适配器
├── operators/      # 算子测试适配器
├── inference/      # 推理测试适配器
├── communication/  # 通信测试适配器
└── common/         # 共享工具
```

### 命名约定

- **适配器文件**: `{framework}_adapter.py`
- **适配器类**: `{Framework}Adapter` (例如 `InfiniCoreAdapter`)
- **测试用例**: `<category>.<framework>.<TestName>`

## 最佳实践

1. **错误处理**: 始终在 try-except 块中包装关键操作
2. **日志记录**: 使用 Python 的 logging 模块进行调试输出
3. **资源管理**: 确保 `teardown()` 正确释放所有资源
4. **配置**: 为所有配置参数提供合理的默认值
5. **文档**: 为所有公共方法添加文档字符串

## 测试更改

1. 创建测试配置文件
2. 使用 `--verbose` 标志运行以获取详细输出
3. 检查输出目录中的 metrics.json
4. 验证日志中的任何错误

```bash
python main.py test_config.json --verbose
```

## 贡献

贡献适配器或指标时：

1. 遵循现有代码风格
2. 为新功能添加文档
3. 包含示例配置
4. 更新相关文档文件

如有问题或讨论，请在 GitHub 上开启 issue。

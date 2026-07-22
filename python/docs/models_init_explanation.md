# `python/aios/models/__init__.py` 源码解释报告

## 文件定位

`models/__init__.py` 是 `aios.models` 包的统一入口，负责导出模型相关 API，并提供模型工厂函数 `create_model`。

## 导入

```python
from .base import BaseLLMModel
from .config import ModelConfig
from .weight import load_weights
```

这三个对象分别代表：

- `BaseLLMModel`：模型抽象基类。
- `ModelConfig`：模型配置结构。
- `load_weights`：checkpoint 权重加载函数。

## `create_model`

```python
def create_model(model_path: str, config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if "qwen3" in model_name:
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(config)
    
    raise ValueError(f"Unsupported model: {model_path}")
```

这是模型工厂函数。

它通过 `model_path.lower()` 判断模型路径或名称中是否包含 `"qwen3"`。如果包含，就延迟导入并创建 `Qwen3ForCausalLM`。

延迟导入的好处是：只有真正创建 Qwen3 模型时才加载 `qwen3.py`。

如果模型名称不匹配，则抛出：

```python
ValueError("Unsupported model: ...")
```

## `__all__`

```python
__all__ = ["BaseLLMModel", "ModelConfig", "load_weights", "create_model"]
```

定义 `aios.models` 的公共 API。

## 扩展方式

如果未来支持其他模型，通常会在这里增加分支：

```python
if "llama" in model_name:
    from .llama import LlamaForCausalLM
    return LlamaForCausalLM(config)
```

同时需要新增对应模型实现文件。

## 总结

`models/__init__.py` 是模型子包的门面和工厂入口。当前只支持 Qwen3，通过 `create_model` 根据模型路径创建 `Qwen3ForCausalLM`，并对外导出配置、基类和权重加载函数。

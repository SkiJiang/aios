# `python/aios/engine/__init__.py` 源码解释报告

## 文件定位

`engine/__init__.py` 是 engine 子包的统一导出入口。

## 导出内容

```python
from .sample import Sampler
from .engine import Engine
from .graph import GraphRunner
```

三个导出对象分别是：

- `Sampler`：logits 到 next token 的采样器。
- `Engine`：GPU 执行资源和 batch forward 的拥有者。
- `GraphRunner`：CUDA graph capture/replay 管理器。

## `__all__`

```python
__all__ = ["Sampler", "Engine", "GraphRunner"]
```

定义 engine 包的公共 API。

## 总结

`engine/__init__.py` 是执行层包入口，集中导出采样、执行和 CUDA graph 三个核心组件。

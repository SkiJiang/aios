# `python/aios/__init__.py` 源码解释报告

## 文件定位

这个文件是 `aios` 包的顶层入口。它决定用户执行 `import aios` 或 `from aios import ...` 时能直接拿到哪些公开对象。

源码很短：

```python
from aios.llm import LLM
from aios.engine import Sampler
from aios.core import SamplingParams, clear_global_ctx

__all__ = ["LLM", "Sampler", "SamplingParams", "clear_global_ctx"]
```

## 导出对象

`LLM` 是用户侧最主要的推理 API，来自 `aios.llm`。普通使用者通常只需要创建：

```python
llm = LLM(model_path)
```

然后调用：

```python
llm.generate(...)
```

`Sampler` 来自 `aios.engine`，用于根据 logits 和采样参数生成下一个 token。

`SamplingParams` 来自 `aios.core`，描述生成配置，例如 `temperature`、`top_k`、`top_p`、`max_tokens`。

`clear_global_ctx` 用于清理全局推理上下文，通常由 `Engine.shutdown()` 间接调用。

## `__all__` 的作用

`__all__` 定义包的公共 API：

```python
__all__ = ["LLM", "Sampler", "SamplingParams", "clear_global_ctx"]
```

当外部代码执行：

```python
from aios import *
```

只会导出这些名字。

## 设计意义

这个入口隐藏了内部目录结构。用户不需要知道 `LLM` 实际定义在 `aios/llm/llm.py`，也不需要知道 `SamplingParams` 定义在 `core.py`。

顶层 API 可以保持稳定，即使后续内部文件拆分或重组，也只需要维护这里的导出。

## 总结

`python/aios/__init__.py` 是 AIOS 包的公共门面，主要导出推理入口 `LLM`、采样器 `Sampler`、采样参数 `SamplingParams` 和上下文清理函数 `clear_global_ctx`。

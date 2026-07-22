# `python/aios/models/base.py` 源码解释报告

## 文件定位

`models/base.py` 定义模型层的抽象基类 `BaseLLMModel`。

```python
class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
```

它连接了两套抽象：

- `ABC`：Python 标准抽象基类机制。
- `BaseOP`：AIOS 自定义权重和子模块管理机制。

## 导入

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aios.layers import BaseOP
```

`TYPE_CHECKING` 下才导入 `torch`，避免运行时为了类型注解额外引入依赖。

## `BaseLLMModel`

`BaseLLMModel` 规定所有顶层 LLM 模型都必须实现：

```python
forward() -> torch.Tensor
```

这里的 `forward` 没有显式参数，因为 AIOS 模型 forward 依赖全局 `Context` 获取当前 batch。

典型实现是 `Qwen3ForCausalLM.forward()`：

```python
hidden_states = self.model.forward()
return self.lm_head.forward(hidden_states)
```

## 为什么继承 `BaseOP`

继承 `BaseOP` 后，顶层模型拥有：

- `state_dict()`
- `load_state_dict()`

这使 `models/weight.py` 可以通过：

```python
for target_name in model.state_dict():
    ...
model.load_state_dict(fused_state_dict)
```

完成 checkpoint 权重加载。

## 总结

`models/base.py` 很小，但它定义了 AIOS 顶层模型的共同接口：模型必须是 `BaseOP`，并且必须提供无参数 `forward()`。运行时输入不通过参数传入，而是通过全局上下文读取。

# `python/aios/layers/__init__.py` 源码解释报告

## 文件定位

`__init__.py` 是 `aios.layers` 包的统一导出入口。

它把 `layers` 目录下各个子模块中的类和函数重新导出，使其他代码可以通过更短、更稳定的路径导入。

例如：

```python
from aios.layers import Linear, RMSNorm, RotaryEmbedding
```

而不需要写成：

```python
from aios.layers.linear import Linear
from aios.layers.norm import RMSNorm
from aios.layers.rotary import RotaryEmbedding
```

## 导入部分

```python
from .activation import silu_and_mul
from .base import BaseOP, StateLessOP, OPList, _concat_prefix
from .linear import Linear, LinearColParallelMerged, LinearQKVMerged
from .norm import RMSNorm, RMSNormFused
from .rotary import RotaryEmbedding
from .attention import apply_rotary_pos_emb, repeat_kv, rotate_half
from .embedding import Embedding, LMHead
```

这些导入可以按职责分组理解。

### activation

```python
from .activation import silu_and_mul
```

导出 MLP 中使用的 fused SwiGLU 激活函数。

### base

```python
from .base import BaseOP, StateLessOP, OPList, _concat_prefix
```

导出项目自定义层系统的基础组件：

- `BaseOP`：有状态算子基类。
- `StateLessOP`：无状态算子基类。
- `OPList`：BaseOP 列表容器。
- `_concat_prefix`：权重 key 拼接辅助函数。

`_concat_prefix` 以下划线开头，但仍然被显式导出。说明它虽然是内部风格命名，但项目内其他模块会使用它，例如 `LMHead.load_state_dict`。

### linear

```python
from .linear import Linear, LinearColParallelMerged, LinearQKVMerged
```

导出基础线性层和 fused projection 变体：

- `Linear`
- `LinearColParallelMerged`
- `LinearQKVMerged`

### norm

```python
from .norm import RMSNorm, RMSNormFused
```

导出 RMSNorm 相关层。

### rotary

```python
from .rotary import RotaryEmbedding
```

导出 RoPE cos/sin cache 生成器。

### attention

```python
from .attention import apply_rotary_pos_emb, repeat_kv, rotate_half
```

导出 attention 相关辅助函数。

### embedding

```python
from .embedding import Embedding, LMHead
```

导出输入 embedding 和输出 lm head。

## `__all__`

```python
__all__ = [
    "silu_and_mul",
    "BaseOP", "StateLessOP", "OPList", "_concat_prefix",
    "Linear", "LinearColParallelMerged", "LinearQKVMerged",
    "RMSNorm", "RMSNormFused", "RotaryEmbedding",
    "apply_rotary_pos_emb", "repeat_kv", "rotate_half",
    "Embedding", "LMHead",
]
```

`__all__` 定义了该包的公开 API 列表。

当用户或模块使用：

```python
from aios.layers import *
```

Python 会只导入 `__all__` 中列出的名字。

同时，它也起到文档作用：告诉读者 `aios.layers` 希望对外暴露哪些符号。

## 为什么需要统一导出

统一导出有几个好处：

### 简化导入路径

模型文件可以写：

```python
from aios.layers import (
    BaseOP,
    Embedding,
    Linear,
    LinearColParallelMerged,
    LinearQKVMerged,
    LMHead,
    OPList,
    RMSNorm,
    RMSNormFused,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    silu_and_mul,
)
```

而不需要分别从多个子模块导入。

### 降低上层对文件结构的耦合

如果将来内部文件重组，只要 `aios.layers` 的导出接口保持不变，上层模型代码可以少改甚至不改。

### 明确 package API

`layers` 目录里可能有内部 helper 或未来新增模块，但只有 `__all__` 里的名字属于当前明确公开的接口。

## 和 Qwen3 模型的关系

`python/aios/models/qwen3.py` 正是通过这个入口导入所有 layer 组件：

```python
from aios.layers import (
    BaseOP,
    Embedding,
    Linear,
    LinearColParallelMerged,
    LinearQKVMerged,
    LMHead,
    OPList,
    RMSNorm,
    RMSNormFused,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    silu_and_mul,
)
```

这说明 `__init__.py` 是模型层和基础 layer 实现之间的 API 边界。

## 导入顺序的影响

当前导入顺序没有复杂副作用，但仍有几个细节：

- `activation.py` 中 `flashinfer` 是函数内部延迟导入，所以导入 `aios.layers` 时不会立刻导入该 flashinfer activation。
- `norm.py` 中 `flashinfer` 也是类初始化时导入，所以导入 `aios.layers` 时不会立刻导入 rmsnorm 算子。
- `rotary.py`、`attention.py`、`embedding.py`、`linear.py` 会在导入 `aios.layers` 时被加载。

因此，`from aios.layers import ...` 的导入成本相对可控。

## 设计特点

这个文件的职责很单一：

- 聚合子模块导出。
- 定义 `aios.layers` 的公共 API。
- 让模型代码保持简洁。

它不包含业务逻辑、不创建对象、不执行权重加载。

## 注意事项

修改这个文件时需要注意：

- 新增 layer 后，如果希望通过 `from aios.layers import X` 使用，需要在这里导入并加入 `__all__`。
- 删除或重命名导出项可能会影响模型文件和外部调用方。
- `_concat_prefix` 虽然以下划线开头，但已经被导出并被其他模块使用，不能简单视为完全私有。
- 避免在这里加入有重副作用的导入，否则会增加 `aios.layers` 的导入成本。

## 总结

`__init__.py` 是 `aios.layers` 的门面文件。它把 activation、base、linear、norm、rotary、attention、embedding 等子模块中的关键类和函数集中导出，为模型代码提供稳定、简洁的导入入口。

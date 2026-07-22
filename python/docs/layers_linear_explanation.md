# `python/aios/layers/linear.py` 源码解释报告

## 文件定位

`linear.py` 定义了 AIOS 推理模型里使用的线性层，以及两个面向大模型结构优化的 fused projection 变体：

- `Linear`：基础全连接层。
- `LinearQKVMerged`：把 attention 的 Q、K、V 三个投影合并成一个大矩阵。
- `LinearColParallelMerged`：把多个输出分支合并成一个大矩阵，当前主要用于 Qwen3 MLP 的 gate/up 投影。

这些类都继承自 `BaseOP`，因此它们的公开 `torch.Tensor` 字段会自动进入项目自定义的 `state_dict/load_state_dict` 权重体系。

## 导入部分

```python
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .base import BaseOP
```

`from __future__ import annotations` 让类型注解延迟求值，减少运行时解析类型的成本，也避免某些循环引用问题。

`Sequence` 用于描述只读序列类型，比 `list` 更泛化，可以接收 list、tuple 等序列。

`torch.nn.functional as F` 提供函数式线性层实现，代码中使用的是 `F.linear`。

`BaseOP` 是项目自己的算子基类，不是 PyTorch 的 `nn.Module`。

## `Linear`

```python
class Linear(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

### 初始化逻辑

`Linear.__init__` 接收三个参数：

- `input_size`：输入特征维度。
- `output_size`：输出特征维度。
- `has_bias`：是否创建 bias，默认不创建。

权重初始化为：

```python
self.weight = torch.empty(output_size, input_size)
```

这里使用 `torch.empty`，表示只分配内存，不填充有效数值。实际权重会在后续加载 checkpoint 时由 `BaseOP.load_state_dict()` 替换。

如果 `has_bias=True`：

```python
self.bias = torch.empty(output_size)
```

如果 `has_bias=False`：

```python
self.bias = None
```

由于 `BaseOP.state_dict()` 只收集公开的 `torch.Tensor` 字段，所以：

- `weight` 一定进入 `state_dict`。
- `bias` 只有在它是 `torch.Tensor` 时才进入 `state_dict`。
- `bias=None` 时会被忽略。

### 前向计算

```python
return F.linear(x, self.weight, self.bias)
```

`F.linear` 的数学形式是：

```text
y = x @ weight.T + bias
```

如果输入 `x` 的最后一维是 `input_size`，输出的最后一维就是 `output_size`。

例如：

```text
x.shape      = [tokens, input_size]
weight.shape = [output_size, input_size]
output.shape = [tokens, output_size]
```

这个类没有继承 `torch.nn.Linear`，原因是项目想自己控制权重存储、加载、融合和命名规则。

## `LinearQKVMerged`

```python
class LinearQKVMerged(Linear):
    """One packed projection for Q, K, and V.

    The packed layout is ``[Q | K | V]`` on the output dimension.  It matches
    mini-sglang's ``LinearQKVMerged`` layout, while remaining replicated on one
    GPU until the tensor-parallel lesson.
    """
```

这个类表示 attention 里的 Q、K、V 三个线性投影被合并成一个投影。

常规 attention 里通常有三个矩阵：

```text
q_proj: hidden_size -> q_size
k_proj: hidden_size -> kv_size
v_proj: hidden_size -> kv_size
```

合并后变成一个矩阵：

```text
qkv_proj: hidden_size -> q_size + kv_size + kv_size
```

布局是：

```text
[Q | K | V]
```

也就是输出维度上按顺序拼接 Q、K、V。

### 初始化逻辑

```python
def __init__(
    self,
    hidden_size: int,
    q_size: int,
    kv_size: int,
    has_bias: bool = False,
) -> None:
    super().__init__(hidden_size, q_size + 2 * kv_size, has_bias)
    self.q_size = q_size
    self.kv_size = kv_size
```

调用父类 `Linear` 时：

```python
input_size = hidden_size
output_size = q_size + 2 * kv_size
```

所以最终权重形状是：

```text
[q_size + 2 * kv_size, hidden_size]
```

`self.q_size` 和 `self.kv_size` 是普通公开字段，但它们是整数，不是 `torch.Tensor`，也不是 `BaseOP`，因此不会进入 `state_dict`。

### 在 Qwen3Attention 中的使用

Qwen3 attention 中会执行：

```python
qkv = self.qkv_proj.forward(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

先用一个大矩阵算出合并结果，再按输出维度切回 Q、K、V。

这样可以减少算子调用次数，也让后续 tensor parallel 版本更容易沿输出列进行切分。

### 和 checkpoint 加载的关系

HuggingFace checkpoint 通常仍然保存为：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

项目在 `python/aios/models/weight.py` 中会把这些源权重按 dim 0 拼接成：

```text
qkv_proj.weight
```

也就是说，`LinearQKVMerged` 本身不关心 HF 的原始 key，只关心合并后的权重形状和前向计算。

## `LinearColParallelMerged`

```python
class LinearColParallelMerged(Linear):
    """One packed column projection for independent output branches.

    For Qwen3 SwiGLU this is ``[gate | up]``.  The name deliberately follows
    mini-sglang so that the later tensor-parallel lesson can shard it without
    changing the model-level contract.
    """
```

这个类用于把多个独立输出分支合并成一个大线性投影。

在 Qwen3 MLP 的 SwiGLU 结构中，通常有两个上投影：

```text
gate_proj: hidden_size -> intermediate_size
up_proj:   hidden_size -> intermediate_size
```

合并后变成：

```text
gate_up_proj: hidden_size -> intermediate_size + intermediate_size
```

输出布局是：

```text
[gate | up]
```

### 初始化逻辑

```python
def __init__(
    self,
    input_size: int,
    output_sizes: Sequence[int],
    has_bias: bool = False,
) -> None:
    self.output_sizes = tuple(output_sizes)
    super().__init__(input_size, sum(self.output_sizes), has_bias)
```

`output_sizes` 表示每个分支的输出大小。

例如：

```python
output_sizes = [intermediate_size, intermediate_size]
```

则：

```python
self.output_sizes = (intermediate_size, intermediate_size)
output_size = sum(self.output_sizes)
```

最终权重形状是：

```text
[sum(output_sizes), input_size]
```

`self.output_sizes` 是 tuple，不会进入 `state_dict`。

### 在 Qwen3MLP 中的使用

Qwen3 MLP 里：

```python
self.gate_up_proj = LinearColParallelMerged(
    config.hidden_size,
    [config.intermediate_size, config.intermediate_size],
)
```

前向时：

```python
gate_up = self.gate_up_proj.forward(x)
return self.down_proj.forward(self._act_fn(gate_up))
```

`self._act_fn` 通常是 `silu_and_mul`。这个 fused activation 会把 `[gate | up]` 拆开，计算：

```text
silu(gate) * up
```

然后交给 `down_proj`。

## 权重命名影响

因为 `Linear` 继承 `BaseOP`，公开 `weight` 字段会被收集。

如果模型里有：

```python
self.qkv_proj = LinearQKVMerged(...)
```

那么权重 key 会是：

```text
qkv_proj.weight
```

如果它嵌套在：

```text
model.layers.0.self_attn
```

完整 key 就会是：

```text
model.layers.0.self_attn.qkv_proj.weight
```

## 设计特点

这个文件的设计重点是推理场景：

- 不做随机初始化，使用 `torch.empty` 等待 checkpoint 覆盖。
- 不继承 `nn.Module`，权重系统由 `BaseOP` 管理。
- fused projection 类只改变权重布局，不改变基础线性计算。
- QKV 和 gate/up 的 checkpoint 适配逻辑放在权重加载器中，不放在线性层里。

## 注意事项

如果新增线性层变体，需要注意：

- 可学习权重必须是公开 `torch.Tensor` 字段，例如 `self.weight`。
- 非权重配置可以是公开非 Tensor 字段，但更建议用 `_` 前缀表示它不参与权重系统。
- 如果需要支持多个分支，前向代码需要明确知道输出布局。
- 如果 checkpoint 中的源权重名字不同，应修改权重加载器，而不是让 layer 直接理解 checkpoint 格式。

## 总结

`linear.py` 是 AIOS 大模型推理层的基础线性投影实现。`Linear` 提供最小的矩阵乘接口，`LinearQKVMerged` 和 `LinearColParallelMerged` 则通过输出维度拼接支持 attention 和 MLP 的 fused 权重布局，为更高效的推理和后续并行切分做准备。

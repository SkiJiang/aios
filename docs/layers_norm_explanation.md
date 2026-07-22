# `python/aios/layers/norm.py` 源码解释报告

## 文件定位

`norm.py` 定义了 RMSNorm 相关层：

- `RMSNorm`：普通 RMSNorm。
- `RMSNormFused`：支持 residual add 与 RMSNorm 融合的版本。

这两个类都继承自 `BaseOP`，因此它们的 `weight` 会自动进入项目自定义的权重系统。

## 导入部分

```python
from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseOP
```

`Tuple` 用于返回值类型注解。

`torch` 用于张量类型和权重创建。

`BaseOP` 是项目自定义的层基类。

## RMSNorm 背景

RMSNorm 是 Root Mean Square Layer Normalization。它和 LayerNorm 类似，但通常不减均值，只根据均方根归一化。

简化数学形式：

```text
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight
```

在大模型推理里，RMSNorm 常用于 Transformer block 前后，例如：

- attention 前的 input layernorm。
- MLP 前的 post attention layernorm。
- 模型最后的 norm。

## `RMSNorm`

```python
class RMSNorm(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
```

### 初始化逻辑

参数：

- `size`：hidden size，也就是最后一维的大小。
- `eps`：数值稳定项，默认 `1e-6`。

权重：

```python
self.weight = torch.empty(size)
```

`weight` 是 RMSNorm 的缩放参数，形状为：

```text
[hidden_size]
```

它是公开 `torch.Tensor` 字段，所以会被 `BaseOP.state_dict()` 收集。

### flashinfer 的延迟导入

```python
from flashinfer import rmsnorm
```

这个导入写在 `__init__` 内部，而不是文件顶部。

这样做的效果是：

- 只有实际创建 `RMSNorm` 实例时才需要导入 `flashinfer`。
- 文件被静态分析或导入时，不会马上要求 `flashinfer` 可用。
- 更贴合 GPU 推理场景，因为 `flashinfer` 是底层高性能算子依赖。

导入后的函数被保存为：

```python
self.rmsnorm = rmsnorm
```

注意这个字段是公开字段，但它不是 `torch.Tensor`，也不是 `BaseOP`，所以不会进入 `state_dict`。

### `forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.rmsnorm(x, self.weight, self.eps)
```

调用 flashinfer 的 `rmsnorm` 算子，返回一个新的归一化结果。

输入输出形状通常保持一致：

```text
x.shape      = [tokens, hidden_size]
output.shape = [tokens, hidden_size]
```

### `forward_inplace`

```python
def forward_inplace(self, x: torch.Tensor) -> None:
    self.rmsnorm(x, self.weight, self.eps, out=x)
```

这个方法使用 `out=x`，把结果写回原张量。

它没有返回值，语义是原地修改输入。

在 attention 中，Q 和 K 会先 reshape 成：

```text
[total_tokens, num_heads, head_dim]
```

然后调用：

```python
self.q_norm.forward_inplace(q)
self.k_norm.forward_inplace(k)
```

这样可以减少额外临时张量分配。

## `RMSNormFused`

```python
class RMSNormFused(BaseOP):
    def __init__(self, size: int, eps: float = 1e-6) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm
```

这个类在普通 RMSNorm 基础上，多引入了：

```python
fused_add_rmsnorm
```

它用于把 residual add 和 RMSNorm 合并到一个底层算子里执行。

### 初始化字段

- `self.eps`：RMSNorm epsilon。
- `self.weight`：可学习缩放参数，会进入 `state_dict`。
- `self.rmsnorm`：普通 RMSNorm flashinfer 函数。
- `self.fused_add_rmsnorm`：融合 residual add 的 flashinfer 函数。

同样，只有 `weight` 是 `torch.Tensor`，因此只有它会作为权重保存和加载。

## `RMSNormFused.forward`

```python
def forward(
    self, x: torch.Tensor, residual: torch.Tensor | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        return self.rmsnorm(x, self.weight, self.eps), x
    self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
    return x, residual
```

这个方法有两种路径。

### 第一层或无 residual

```python
if residual is None:
    return self.rmsnorm(x, self.weight, self.eps), x
```

当 residual 不存在时，只做 RMSNorm，并把原始 `x` 作为 residual 返回。

返回值是：

```text
(normalized_x, residual_x)
```

其中：

- `normalized_x` 用于进入 attention 或 MLP。
- `residual_x` 保存残差路径。

### 有 residual 时

```python
self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
return x, residual
```

当 residual 已存在时，调用 fused 算子。

它通常会完成类似逻辑：

```text
residual = residual + x
x = rmsnorm(residual)
```

具体是否原地写入 `x` 和 `residual` 取决于 flashinfer 算子的实现，但从函数调用和返回方式看，代码预期它会原地更新传入张量。

这就是为什么调用后直接返回：

```python
return x, residual
```

## 在 Qwen3DecoderLayer 中的使用

Qwen3 decoder layer 中：

```python
self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
self.post_attention_layernorm = RMSNormFused(
    config.hidden_size, config.rms_norm_eps
)
```

前向过程：

```python
hidden_states, residual = self.input_layernorm.forward(hidden_states, residual)
hidden_states = self.self_attn.forward(hidden_states, position_embeddings)
hidden_states, residual = self.post_attention_layernorm.forward(
    hidden_states, residual
)
hidden_states = self.mlp.forward(hidden_states)
```

这体现了 pre-norm Transformer block 的流程：

1. 对输入做 norm。
2. 进入 attention。
3. attention 输出和 residual 融合，再 norm。
4. 进入 MLP。

## 在 Qwen3Attention 中的使用

Attention 中还会创建：

```python
self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
```

这用于 Qwen3 的 Q/K norm：

```python
self.q_norm.forward_inplace(q)
self.k_norm.forward_inplace(k)
```

这里的 norm size 是 `head_dim`，不是 `hidden_size`。

## 权重命名

因为 `weight` 是公开 Tensor 字段，所以会进入 `state_dict`。

例如：

```text
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
model.layers.0.self_attn.q_norm.weight
model.layers.0.self_attn.k_norm.weight
model.norm.weight
```

这些 key 会由 `BaseOP.state_dict()` 递归生成。

## 设计特点

这个文件的设计目标是推理性能：

- 使用 flashinfer 提供高性能 RMSNorm 算子。
- 提供 inplace Q/K norm，减少内存分配。
- 提供 fused add RMSNorm，减少 residual add 和 norm 的算子调度成本。
- 权重只保留必要的 `weight`，其余运行时函数和配置不作为 checkpoint 状态。

## 注意事项

使用或修改时需要注意：

- `flashinfer` 必须在运行环境中可用，否则实例化会失败。
- `forward_inplace` 会修改输入张量，调用方不能假设输入保持不变。
- `RMSNormFused.forward` 的返回值始终是二元组 `(x, residual)`。
- `weight` 使用 `torch.empty`，必须经过 checkpoint 加载后才有有效数值。
- 如果未来要让 `eps` 不进入 `state_dict`，当前实现已经满足，因为它不是 Tensor。

## 总结

`norm.py` 提供了 AIOS Transformer 推理中使用的 RMSNorm 实现。`RMSNorm` 负责普通归一化和原地归一化，`RMSNormFused` 则将 residual add 和 RMSNorm 融合，减少推理阶段的内存和算子开销。

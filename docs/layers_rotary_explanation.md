# `python/aios/layers/rotary.py` 源码解释报告

## 文件定位

`rotary.py` 定义了 `RotaryEmbedding`，用于生成 RoPE 位置编码所需的 cos/sin 缓存。

RoPE，全称 Rotary Position Embedding，是 Qwen、LLaMA 等大模型 attention 中常用的位置编码方法。它不会把位置向量直接加到 hidden state 上，而是在 attention 的 Q/K 向量上进行旋转变换。

这个文件只负责生成和读取 cos/sin，不负责真正把 RoPE 应用到 Q/K 上。真正的旋转逻辑在 `attention.py` 的 `apply_rotary_pos_emb` 中。

## 导入部分

```python
from __future__ import annotations

import torch

from .base import StateLessOP
```

`RotaryEmbedding` 继承自 `StateLessOP`，表示它没有需要从 checkpoint 加载的可学习权重。

虽然它内部保存了 `_cos_cache` 和 `_sin_cache` 两个 Tensor，但它们是预计算缓存，不是模型参数。

## `RotaryEmbedding`

```python
class RotaryEmbedding(StateLessOP):
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float = 1000000.0):
        super().__init__()
```

### 参数含义

- `head_dim`：每个 attention head 的维度。
- `max_position_embeddings`：最大位置长度，也就是最多预计算多少个 position。
- `base`：RoPE 的频率基数，默认 `1000000.0`。

Qwen 系列模型通常会通过配置传入 `rope_theta`，对应这里的 `base`。

## CPU 上强制创建缓存

```python
with torch.device("cpu"):
```

注释说明：

```python
# Force CPU creation even inside `with torch.device("meta")` context,
# since cos/sin caches are precomputed buffers (not learned parameters).
```

这段设计很重要。

项目创建模型时可能处于：

```python
with torch.device("meta"):
```

这样的上下文中。`meta` device 常用于只构建模型结构而不分配真实内存。

但是 RoPE 的 cos/sin cache 不是可学习权重，不需要等 checkpoint 加载。它们可以直接计算出来，而且后续可以通过 `set_device()` 移到目标设备。

所以这里显式使用 CPU 创建，避免被外层 `meta` 上下文影响。

## 频率向量计算

```python
inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
```

这里计算 RoPE 的 inverse frequency。

`torch.arange(0, head_dim, 2)` 生成偶数维度索引：

```text
0, 2, 4, ..., head_dim - 2
```

然后除以 `head_dim`：

```text
[0/head_dim, 2/head_dim, 4/head_dim, ...]
```

再以 `base` 为底做指数：

```text
base ** (...)
```

最后取倒数。

得到的 `inv_freq` 形状是：

```text
[head_dim / 2]
```

## 位置向量和频率外积

```python
t = torch.arange(max_position_embeddings, dtype=torch.float32)
freqs = torch.outer(t, inv_freq)
```

`t` 表示所有位置：

```text
0, 1, 2, ..., max_position_embeddings - 1
```

`torch.outer(t, inv_freq)` 计算每个 position 和每个 frequency 的乘积。

形状：

```text
t.shape        = [max_position_embeddings]
inv_freq.shape = [head_dim / 2]
freqs.shape    = [max_position_embeddings, head_dim / 2]
```

## 扩展到完整 head_dim

```python
emb = torch.cat((freqs, freqs), dim=-1)
```

把 `freqs` 在最后一维复制一份，得到：

```text
emb.shape = [max_position_embeddings, head_dim]
```

这样后续可以直接和 Q/K 的最后一维对齐。

## 预计算 cos/sin cache

```python
self._cos_cache = emb.cos()  # (max_pos, head_dim)
self._sin_cache = emb.sin()  # (max_pos, head_dim)
```

这两个字段名都以下划线开头，因此不会进入 `BaseOP.state_dict()`。

它们的形状都是：

```text
[max_position_embeddings, head_dim]
```

这是典型的缓存数据：

- 由配置决定。
- 可以重新计算。
- 不从 checkpoint 加载。
- 不作为模型可学习参数。

## `forward`

```python
def forward(self, position_ids: torch.Tensor):
    # position_ids: (batch, seq_len)
    cos = self._cos_cache[position_ids]  # (batch, seq_len, head_dim)
    sin = self._sin_cache[position_ids]  # (batch, seq_len, head_dim)
    return cos.unsqueeze(1), sin.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
```

### 输入

`position_ids` 是每个 token 的位置 id。

注释中写的是：

```text
[batch, seq_len]
```

在连续 batching 或 packed token 推理中，具体形状可能由 `ctx.batch.positions` 决定，但语义都是按位置索引 cos/sin。

### 查表

```python
cos = self._cos_cache[position_ids]
sin = self._sin_cache[position_ids]
```

根据 position id 从预计算缓存中取对应位置。

如果 `position_ids.shape = [batch, seq_len]`，则：

```text
cos.shape = [batch, seq_len, head_dim]
sin.shape = [batch, seq_len, head_dim]
```

### 增加 head broadcast 维度

```python
return cos.unsqueeze(1), sin.unsqueeze(1)
```

返回形状变成：

```text
[batch, 1, seq_len, head_dim]
```

中间的 `1` 用于和 Q/K 的 head 维度广播。

典型 attention 张量形状可能是：

```text
q.shape = [batch, num_heads, seq_len, head_dim]
k.shape = [batch, num_kv_heads, seq_len, head_dim]
```

cos/sin 的 head 维度是 1，因此可以广播到不同 head 数量。

## `set_device`

```python
def set_device(self, device: torch.device):
    self._cos_cache = self._cos_cache.to(device)
    self._sin_cache = self._sin_cache.to(device)
```

由于缓存强制在 CPU 创建，实际推理前需要把它们移动到 GPU 或其他目标设备。

`set_device` 就负责这件事。

它会替换原来的 cache Tensor：

```python
self._cos_cache = ...
self._sin_cache = ...
```

## 和 `StateLessOP` 的关系

`RotaryEmbedding` 继承 `StateLessOP`：

```python
class RotaryEmbedding(StateLessOP):
```

所以：

```python
state_dict() -> {}
```

这和字段命名也是一致的：

```python
self._cos_cache
self._sin_cache
```

两者都不会出现在 checkpoint 权重中。

## 在 Qwen3Model 中的使用

模型中会创建：

```python
self._rotary_emb = RotaryEmbedding(
    config.head_dim,
    config.max_position_embeddings,
    config.rope_theta,
)
```

字段名是 `_rotary_emb`，因此即使它是 `BaseOP`，也不会被 `BaseOP.state_dict()` 递归收集。

前向时：

```python
position_embeddings = self._rotary_emb.forward(batch.positions)
```

然后传入每一层 decoder layer，再由 attention 应用到 Q/K 上。

## 设计特点

这个文件的设计重点是：

- RoPE cache 预计算，避免每步重复计算三角函数。
- cache 不进入 checkpoint，符合 `StateLessOP` 语义。
- 强制 CPU 创建，避免 meta device 上下文导致缓存不可用。
- 通过 `set_device` 明确迁移到目标设备。

## 注意事项

使用时需要注意：

- `position_ids` 不能超过 `max_position_embeddings - 1`。
- 如果模型迁移到 GPU，必须确保调用 `set_device` 或等价逻辑。
- `_cos_cache/_sin_cache` 是缓存，不是权重，不能依赖 checkpoint 恢复。
- `head_dim` 通常需要是偶数，因为 RoPE 使用一半维度生成频率再拼接。

## 总结

`rotary.py` 负责为 RoPE 位置编码准备 cos/sin 缓存。它不参与权重加载，而是根据模型配置预计算位置频率，并在前向时按 position id 查表，为 attention 中的 Q/K 旋转提供输入。

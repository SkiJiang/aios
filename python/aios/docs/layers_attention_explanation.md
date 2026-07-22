# `python/aios/layers/attention.py` 源码解释报告

## 文件定位

`attention.py` 提供 attention 相关的基础张量变换函数：

- `rotate_half`：RoPE 旋转中的半维度变换。
- `apply_rotary_pos_emb`：把 RoPE cos/sin 应用到 Q/K。
- `repeat_kv`：把 K/V head 重复到和 Q head 对齐，常用于 grouped-query attention。

这个文件没有定义 `BaseOP` 子类，也没有模型权重。它更像是 attention 计算中的函数工具层。

## 导入部分

```python
from __future__ import annotations

import torch
```

这里只依赖 `torch`。

所有函数都直接操作 `torch.Tensor`，没有状态，也不需要 checkpoint。

## `rotate_half`

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
```

这个函数用于 RoPE 旋转公式。

### 输入拆分

```python
x1, x2 = x.chunk(2, dim=-1)
```

沿最后一维把 `x` 平分成两部分：

```text
x = [x1 | x2]
```

如果：

```text
x.shape = [..., head_dim]
```

那么：

```text
x1.shape = [..., head_dim / 2]
x2.shape = [..., head_dim / 2]
```

这要求最后一维通常是偶数。

### 旋转半边

```python
return torch.cat((-x2, x1), dim=-1)
```

返回：

```text
[-x2 | x1]
```

这个变换相当于在二维平面里做 90 度方向的旋转配套变换，是 RoPE 公式中的核心组成。

## `apply_rotary_pos_emb`

```python
def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

这个函数把 RoPE 应用到 query 和 key。

### 输入

- `q`：query tensor。
- `k`：key tensor。
- `cos`：由 `RotaryEmbedding` 生成的 cos cache。
- `sin`：由 `RotaryEmbedding` 生成的 sin cache。

常见形状：

```text
q.shape   = [batch, num_heads, seq_len, head_dim]
k.shape   = [batch, num_kv_heads, seq_len, head_dim]
cos.shape = [batch, 1, seq_len, head_dim]
sin.shape = [batch, 1, seq_len, head_dim]
```

在本项目的 packed token 推理路径中，Q/K 也可能是按 total tokens 展平后的形状。只要 `cos/sin` 能通过广播匹配 Q/K，公式就能成立。

### dtype 对齐

```python
cos = cos.to(q.dtype)
sin = sin.to(q.dtype)
```

RoPE cache 在 `rotary.py` 中用 `float32` 创建。

实际推理中的 Q/K 可能是：

- `float16`
- `bfloat16`
- 其他低精度 dtype

这里把 cos/sin 转成和 q 一样的 dtype，避免混合精度下产生不必要的类型提升或 kernel 不匹配。

注意这里按 `q.dtype` 对齐，没有单独按 `k.dtype` 对齐。通常 Q/K dtype 一致。

### RoPE 公式

```python
q * cos + rotate_half(q) * sin
k * cos + rotate_half(k) * sin
```

这就是 RoPE 的核心计算。

对每个位置上的向量，使用该位置对应的 cos/sin 对 Q/K 做旋转。

返回：

```python
(rotated_q, rotated_k)
```

输出形状和输入 Q/K 保持一致。

## `repeat_kv`

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)
```

这个函数用于 grouped-query attention 或 multi-query attention。

在这类结构中：

```text
num_q_heads > num_kv_heads
```

多个 query head 会共享同一组 key/value head。

为了使用某些普通 attention 计算路径，需要把 K/V head 重复展开，使它们的 head 数和 Q 对齐。

### 快速路径

```python
if n_rep == 1:
    return x
```

如果不需要重复，直接返回原张量，避免无意义的 view/expand/reshape。

### 输入形状

```python
b, h, s, d = x.shape
```

要求输入是四维：

```text
[batch, num_kv_heads, seq_len, head_dim]
```

其中：

- `b`：batch size。
- `h`：KV head 数。
- `s`：sequence length。
- `d`：head dim。

### 插入重复维度

```python
x[:, :, None, :, :]
```

形状从：

```text
[b, h, s, d]
```

变成：

```text
[b, h, 1, s, d]
```

新增的维度用于重复。

### expand 逻辑

```python
.expand(b, h, n_rep, s, d)
```

`expand` 不会真实复制数据，而是通过 stride 视图进行广播。

形状变成：

```text
[b, h, n_rep, s, d]
```

### reshape 合并 head 维度

```python
.reshape(b, h * n_rep, s, d)
```

最后把 `h` 和 `n_rep` 合并：

```text
[b, h * n_rep, s, d]
```

如果：

```text
h = num_kv_heads
n_rep = num_q_heads / num_kv_heads
```

那么输出 head 数就是：

```text
num_q_heads
```

## 和 Qwen3Attention 的关系

Qwen3 attention 中，RoPE 应用逻辑类似：

```python
cos, sin = position_embeddings
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

`repeat_kv` 在当前 Qwen3 路径中不一定直接调用，因为项目可能使用支持 GQA 的 attention backend。但这个函数作为通用辅助函数被导出，方便其他 attention 实现或 fallback 路径使用。

## 设计特点

这个文件的设计很轻量：

- 只包含无状态函数。
- 不依赖项目的 `BaseOP`。
- 不包含 checkpoint 逻辑。
- 函数粒度小，便于复用。
- RoPE 和 KV repeat 被拆开，避免 attention 主类里堆积底层张量操作。

## 注意事项

使用这些函数时需要注意：

- `rotate_half` 要求最后一维可以平均分成两半。
- `apply_rotary_pos_emb` 要求 `cos/sin` 能广播到 `q/k` 的形状。
- `apply_rotary_pos_emb` 默认 Q/K dtype 一致。
- `repeat_kv` 要求输入是四维 `[batch, heads, seq_len, head_dim]`。
- `repeat_kv` 使用 `expand` 后再 `reshape`，可能在某些情况下触发实际拷贝，但语义上避免了手动 `repeat` 的直接复制。

## 总结

`attention.py` 是 attention 张量操作的辅助函数集合。它实现了 RoPE 所需的半维旋转、Q/K 位置旋转，以及 GQA/MQA 中常见的 KV head 重复逻辑，为上层 attention 模块提供简单、可复用的基础工具。

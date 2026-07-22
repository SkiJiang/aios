# `python/aios/layers/embedding.py` 源码解释报告

## 文件定位

`embedding.py` 定义了语言模型中两个和词表相关的层：

- `Embedding`：输入 token id 到 hidden state 的词嵌入层。
- `LMHead`：hidden state 到 vocabulary logits 的输出投影层。

这两个类都继承自 `BaseOP`，因此权重加载和导出遵循项目自定义的 `state_dict/load_state_dict` 规则。

## 导入部分

```python
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from aios.core import get_global_ctx

from .base import BaseOP, _concat_prefix
```

`Dict` 用于 `load_state_dict` 和 `state_dict` 的类型注解。

`torch.nn.functional as F` 提供函数式 embedding 和 linear 操作。

`get_global_ctx` 用于获取当前推理上下文，`LMHead` 会根据 batch 状态决定是否只计算最后一个 token 的 logits。

`BaseOP` 提供权重收集和加载能力。

`_concat_prefix` 用于生成 `lm_head.weight` 这类层级 key。

## `Embedding`

```python
class Embedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = torch.empty(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)
```

### 初始化逻辑

`Embedding` 接收两个参数：

- `num_embeddings`：词表大小，也就是 vocabulary size。
- `embedding_dim`：每个 token 的向量维度，也就是 hidden size。

权重形状是：

```text
[num_embeddings, embedding_dim]
```

例如 vocab size 是 151936，hidden size 是 4096，则：

```text
weight.shape = [151936, 4096]
```

这里使用 `torch.empty`，只分配内存，不初始化有效值。真实权重由 checkpoint 加载覆盖。

由于字段名是 `weight`，不是 `_weight`，且类型是 `torch.Tensor`，所以它会被 `BaseOP.state_dict()` 收集。

### 前向计算

```python
return F.embedding(input_ids, self.weight)
```

`input_ids` 是 token id 张量，通常形状可能是：

```text
[total_tokens]
```

或者更一般的整数张量。

输出形状是在输入 shape 后面追加 `embedding_dim`：

```text
input_ids.shape = [total_tokens]
output.shape    = [total_tokens, embedding_dim]
```

`F.embedding` 本质上是按 token id 查表。

## `LMHead`

```python
class LMHead(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tie_word_embeddings: bool = False,
        tied_embedding: Embedding | None = None,
    ):
        self._tie_word_embeddings = tie_word_embeddings
        self._tied_embedding = tied_embedding
        if not tie_word_embeddings:
            self.weight = torch.empty(num_embeddings, embedding_dim)
```

`LMHead` 是语言模型最后的输出层，把 hidden state 投影到词表维度，得到 logits。

### 参数含义

- `num_embeddings`：词表大小。
- `embedding_dim`：hidden size。
- `tie_word_embeddings`：是否和输入 embedding 共享权重。
- `tied_embedding`：当共享权重时，指向输入 `Embedding` 实例。

### 权重共享逻辑

如果 `tie_word_embeddings=False`：

```python
self.weight = torch.empty(num_embeddings, embedding_dim)
```

`LMHead` 有自己的输出权重，会进入 `state_dict`。

如果 `tie_word_embeddings=True`：

```python
self._tied_embedding = tied_embedding
```

此时不会创建 `self.weight`，前向时直接使用输入 embedding 的 `weight`。

注意字段名前缀：

- `_tie_word_embeddings` 以下划线开头，不进入 `state_dict`。
- `_tied_embedding` 以下划线开头，不作为子模块递归收集。

这很重要：如果 `tied_embedding` 是公开字段，`BaseOP` 可能会递归收集它，导致权重 key 重复或结构不符合预期。

## `LMHead.forward`

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ctx = get_global_ctx()
    batch = ctx.batch
    if batch.is_prefill:
        indices = batch.attn_metadata.get_last_indices(batch.size)
        x = x[indices].contiguous()
    w = self._tied_embedding.weight if self._tie_word_embeddings else self.weight
    return F.linear(x, w)
```

### 获取全局上下文

```python
ctx = get_global_ctx()
batch = ctx.batch
```

项目通过全局上下文保存当前推理 batch、attention metadata、backend 等运行时信息。

`LMHead` 需要知道当前是不是 prefill 阶段。

### prefill 阶段只取最后 token

```python
if batch.is_prefill:
    indices = batch.attn_metadata.get_last_indices(batch.size)
    x = x[indices].contiguous()
```

在大模型推理中：

- prefill 阶段会一次处理 prompt 的多个 token。
- decode 阶段通常一次只处理一个新 token。

最终采样下一个 token 时，只需要每条序列最后一个 token 的 logits。

所以在 prefill 阶段，`LMHead` 会通过 `get_last_indices` 找到每个请求最后 token 的 hidden state，只对这些位置计算 logits，避免对 prompt 中所有 token 都计算完整词表输出。

`contiguous()` 保证被索引后的张量内存连续，便于后续线性计算。

### 选择输出权重

```python
w = self._tied_embedding.weight if self._tie_word_embeddings else self.weight
```

如果启用词嵌入共享，就使用输入 embedding 的权重。

否则使用 `LMHead` 自己的 `weight`。

### 输出 logits

```python
return F.linear(x, w)
```

`F.linear` 会计算：

```text
logits = x @ w.T
```

如果：

```text
x.shape = [batch_size, hidden_size]
w.shape = [vocab_size, hidden_size]
```

则：

```text
logits.shape = [batch_size, vocab_size]
```

## `LMHead.load_state_dict`

```python
def load_state_dict(
    self,
    state_dict: Dict[str, torch.Tensor],
    *,
    prefix: str = "",
    _internal: bool = False,
) -> None:
    if self._tie_word_embeddings:
        # Pop lm_head.weight if present (tied to embedding)
        key = _concat_prefix(prefix, "weight")
        if key in state_dict:
            state_dict.pop(key)
    else:
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
```

这个方法覆盖了 `BaseOP.load_state_dict()`，目的是处理 tied embedding 的特殊情况。

### 未共享权重

如果 `self._tie_word_embeddings=False`，直接调用父类加载：

```python
super().load_state_dict(...)
```

此时会加载：

```text
lm_head.weight
```

### 共享权重

如果 `self._tie_word_embeddings=True`，`LMHead` 不应该再加载自己的权重，因为它使用的是 `Embedding.weight`。

但有些 checkpoint 里可能仍然包含：

```text
lm_head.weight
```

所以这里做了兼容：

```python
key = _concat_prefix(prefix, "weight")
if key in state_dict:
    state_dict.pop(key)
```

如果存在就从字典中删掉，避免最外层 `load_state_dict` 报 unexpected key。

## `LMHead.state_dict`

```python
def state_dict(
    self,
    *,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    if self._tie_word_embeddings:
        return {}
    return super().state_dict(prefix=prefix)
```

如果共享权重，`LMHead` 不导出任何权重。

如果不共享权重，使用 `BaseOP.state_dict()` 导出自己的 `weight`。

这个逻辑和 `load_state_dict` 对应：

- tied embedding：权重只在输入 embedding 中出现。
- untied embedding：`LMHead` 有独立 `lm_head.weight`。

## 在 Qwen3ForCausalLM 中的使用

模型顶层会创建：

```python
self.model = Qwen3Model(config)
self.lm_head = LMHead(
    config.vocab_size,
    config.hidden_size,
    tie_word_embeddings=config.tie_word_embeddings,
    tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
)
```

如果配置开启 `tie_word_embeddings`，`lm_head` 会复用 `model.embed_tokens.weight`。

如果没有开启，则 `lm_head` 有自己的输出权重。

## 设计特点

这个文件体现了两个推理优化点：

- 词嵌入和输出头支持共享权重，减少显存占用。
- prefill 阶段只计算最后 token logits，避免对全部 prompt token 做词表投影。

同时，它把 checkpoint 特殊情况封装在 `LMHead` 内部，让上层模型不用关心 `lm_head.weight` 是否应该存在。

## 注意事项

新增或修改这类层时需要注意：

- 共享引用字段应以下划线开头，避免被 `BaseOP` 递归收集。
- 如果 `tie_word_embeddings=True`，必须保证 `tied_embedding` 不为 `None`，否则前向时访问 `.weight` 会失败。
- `LMHead.forward` 依赖全局上下文，不能脱离 AIOS 推理流程随意调用，除非先设置好 global context。
- `load_state_dict` 会修改传入的 `state_dict`，这是 `BaseOP` 权重加载体系的既有约定。

## 总结

`embedding.py` 负责模型的输入查表和输出词表投影。`Embedding` 是简单的 token embedding 层；`LMHead` 在此基础上处理了 tied embedding、prefill 阶段 logits 裁剪、以及 checkpoint 中可选 `lm_head.weight` 的兼容逻辑。

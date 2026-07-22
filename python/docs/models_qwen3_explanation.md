# `python/aios/models/qwen3.py` 源码解释报告

## 文件定位

`qwen3.py` 定义 AIOS 当前支持的 Qwen3 Causal LM 模型结构。

主要类：

- `Qwen3Attention`
- `Qwen3MLP`
- `Qwen3DecoderLayer`
- `Qwen3Model`
- `Qwen3ForCausalLM`

这些类使用 `BaseOP` 体系，而不是 PyTorch `nn.Module`。

## `Qwen3Attention`

初始化阶段保存 attention 配置：

```text
num_heads
num_kv_heads
head_dim
q_size
kv_size
```

并创建子层：

- `qkv_proj`：合并的 Q/K/V 投影，输出布局 `[Q | K | V]`。
- `o_proj`：attention 输出投影。
- `q_norm`：Q head 上的 RMSNorm。
- `k_norm`：K head 上的 RMSNorm。

`_scale` 和 `_layer_idx` 以下划线开头，不进入 `state_dict`。其中 `_layer_idx` 用于告诉 attention backend 当前是第几层，从而写入对应层的 KV cache。

### forward 流程

```text
hidden_states
-> qkv_proj
-> split 为 q/k/v
-> reshape 为 head 形状
-> q_norm/k_norm
-> apply_rotary_pos_emb
-> ctx.attn_backend.forward
-> o_proj
```

关键点是 attention 计算不在模型文件中手写，而是交给：

```python
ctx.attn_backend.forward(q, k, v, self._layer_idx, ctx.batch)
```

这让模型结构和具体 attention backend 解耦。

## `Qwen3MLP`

初始化：

```python
self.gate_up_proj = LinearColParallelMerged(...)
self.down_proj = Linear(...)
```

`gate_up_proj` 把 gate 和 up 两个分支合并成一次投影：

```text
[gate | up]
```

激活函数只支持：

```python
case "silu":
    self._act_fn = silu_and_mul
```

如果配置不是 `"silu"`，抛 `ValueError`。

forward：

```text
x
-> gate_up_proj
-> silu_and_mul
-> down_proj
```

其中 `silu_and_mul` 语义是：

```text
silu(gate) * up
```

## `Qwen3DecoderLayer`

一个 decoder layer 包含：

- `self_attn`
- `mlp`
- `input_layernorm`
- `post_attention_layernorm`

两个 norm 都使用 `RMSNormFused`，用于融合 residual add 和 norm。

forward 顺序：

```text
input_layernorm
-> self_attn
-> post_attention_layernorm
-> mlp
```

返回：

```python
(hidden_states, residual)
```

这里 residual 会跨 attention 和 MLP 子层传递。

## `Qwen3Model`

初始化：

```python
self.embed_tokens = Embedding(...)
self.layers = OPList([...])
self.norm = RMSNormFused(...)
self._rotary_emb = RotaryEmbedding(...)
```

`OPList` 用于让 decoder layer 列表能被 `BaseOP.state_dict()` 递归遍历。

`_rotary_emb` 以下划线开头，不进入权重系统，因为 RoPE cache 不是 checkpoint 权重。

forward：

```text
ctx.batch.input_ids
-> embed_tokens
-> rotary_emb(batch.positions)
-> 逐层 decoder
-> final norm
```

模型输入来自全局 context，而不是 forward 参数。

## `Qwen3ForCausalLM`

顶层 causal LM：

```python
self.model = Qwen3Model(config)
self.lm_head = LMHead(...)
```

如果 `tie_word_embeddings=True`，`LMHead` 复用 `self.model.embed_tokens.weight`。

forward：

```text
Qwen3Model.forward()
-> LMHead.forward()
-> logits
```

输出 logits 后，`Engine.forward_batch()` 会调用 `Sampler` 得到 next token。

## 权重命名

由于所有模块都是 `BaseOP`，权重 key 由字段名递归生成。例如：

```text
model.embed_tokens.weight
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_up_proj.weight
model.norm.weight
lm_head.weight
```

`models/weight.py` 会按这些目标 key 读取或拼接 checkpoint 权重。

## 总结

`qwen3.py` 是 AIOS 的模型结构核心。它定义 Qwen3 的 attention、MLP、decoder layer、主体模型和 causal LM 输出头，同时把底层 attention 计算委托给全局 context 中的 attention backend，使模型结构与 FlashInfer/KV cache 实现解耦。

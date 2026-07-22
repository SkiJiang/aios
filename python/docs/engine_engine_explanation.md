# `python/aios/engine/engine.py` 源码解释报告

## 文件定位

`engine.py` 定义 `Engine`，它是 AIOS 的 GPU 执行资源拥有者。

它负责：

- 设置 CUDA device 和 stream。
- 创建模型并加载权重。
- 分配 KV cache。
- 创建全局 `Context`。
- 创建 FlashInfer attention backend。
- 可选创建 CUDA graph runner。
- 执行一次 batch forward 并采样 next token。

## 初始化流程

`Engine.__init__` 接收模型路径、模型配置、dtype、device、并发上限、显存比例和 CUDA graph 参数。

### CUDA stream

```python
torch.cuda.set_device(device)
self.stream = torch.cuda.Stream(device=device)
torch.cuda.set_stream(self.stream)
```

Engine 使用自己的 CUDA stream 执行推理。

### 创建模型和加载权重

```python
with torch.device("meta"):
    self.model = create_model(model_path, model_config)
load_weights(self.model, model_path, device, dtype)
self.model.model._rotary_emb.set_device(device)
```

模型先在 `meta` device 上创建，避免初始化阶段分配真实权重内存。

随后 `load_weights` 从 checkpoint 读取真实权重并放到目标 GPU。

RoPE cache 不是 checkpoint 权重，需要单独移动到 device。

### 计算 KV cache 容量

```python
self.num_pages = self._determine_num_pages(...)
self.max_seq_len = min(model_config.max_position_embeddings, self.num_pages)
self.aligned_max_seq_len = _align_up_32(self.max_seq_len)
```

`_determine_num_pages` 根据加载模型前后的空闲显存估算剩余显存能容纳多少 KV page。

### 创建 Context 和 KV cache

```python
self.ctx = Context(page_size=1)
self.ctx.kv_cache = self.kv_cache = MHAKVCache(...)
self.ctx.page_table = self.page_table = torch.zeros(...)
set_global_ctx(self.ctx)
```

全局 context 会被模型 forward 和 attention backend 使用。

### 创建 attention backend

```python
self.ctx.attn_backend = self.attn_backend = FlashInferBackend(model_config)
```

模型中的 `Qwen3Attention` 会通过 context 调用它。

### dummy request

`dummy_req` 和 `dummy_page` 用于 CUDA graph batch padding。

dummy request 占用最后一个 table row，dummy page 是额外分配的 page。

## `reset_page_table`

```python
self.page_table[: self.max_running_reqs].zero_()
self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
```

真实请求行清零。

dummy 行填满 dummy page，保证 padding 请求有合法 page table。

## `_sync_get_free_memory`

同步 CUDA，清理 cache，重置峰值统计，然后返回当前空闲显存。

用于模型加载前后估算模型显存占用。

## `_determine_num_pages`

计算：

```text
model_memory = initial_free_memory - free_after_model
cache_per_page = 2 * head_dim * num_kv_heads * dtype.itemsize * num_layers
available_memory = memory_ratio * initial_free_memory - model_memory
num_pages = available_memory // cache_per_page
```

这里的 `2` 对应 K 和 V。

如果 page 数不足，抛断言错误。

## `forward_batch`

这是一次 batch 推理入口。

流程：

1. 确认当前 CUDA stream 是 Engine stream。
2. 进入 `ctx.forward_batch(batch)`。
3. 如果可用 CUDA graph，调用 `graph_runner.replay(batch)`。
4. 否则调用 `self.model.forward()`。
5. 对 batch 中每个真实请求调用 `req.complete_one()`。
6. 去掉 padding logits。
7. 对每个请求用 `Sampler` 采样 next token。
8. 返回 int32 next token tensor。

模型 forward 本身不接收 batch 参数，依赖全局 context。

## `shutdown`

如果存在 graph runner，销毁 CUDA graph 相关结构。

最后调用：

```python
clear_global_ctx()
```

避免全局 context 泄漏。

## `_align_up_32`

```python
return (num + 31) // 32 * 32
```

把最大序列长度向上对齐到 32 的倍数，便于 page table 和 CUDA graph buffer 使用。

## 总结

`Engine` 是 AIOS 的执行核心。它拥有 GPU 资源、模型、KV cache、attention backend 和可选 CUDA graph runner，并提供 `forward_batch` 把一个 scheduler batch 转成 next token。

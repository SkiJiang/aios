# `python/aios/attention/fi.py` 源码解释报告

## 文件定位

`attention/fi.py` 实现基于 FlashInfer 的 attention backend，是 AIOS 当前真实执行 attention 的核心模块。

它负责：

- 创建 FlashInfer prefill/decode wrapper。
- 根据 batch 构建 FlashInfer metadata。
- 把当前 K/V 写入 KV cache。
- 调用 FlashInfer wrapper 运行 paged KV attention。
- 支持 CUDA graph capture/replay。

## `_next_power_of_2`

```python
def _next_power_of_2(n: int) -> int:
```

返回大于等于 `n` 的 2 的幂。

它用于扩展 `cached_ones_cpu`，避免每次 batch size 变化都重新分配刚好大小的 pinned CPU tensor。

## `FICaptureData`

```python
@dataclass
class FICaptureData(BaseCaptureData):
```

这是 FlashInfer CUDA graph capture 使用的数据容器。

它提供两个属性：

```python
one_tensor -> self.seq_lens
indices -> self.page_table
```

FlashInfer graph wrapper 需要 `last_page_len_buffer` 和 `indices_buffer`，这里复用基础 capture buffer 中的张量。

## `FIMetadata`

`FIMetadata` 保存一次 FlashInfer attention plan 需要的全部参数。

重要字段：

- `cu_seqlens_q_cpu`：query 序列 cumulative length，CPU pinned。
- `cu_seqlens_k_cpu`：key 序列 cumulative length，CPU pinned。
- `cu_seqlens_q_gpu`：GPU 上的 query cumulative length，用于取最后 token index。
- `indices`：paged KV indices，CUDA tensor。
- `last_page_len_cpu`：每条请求最后页长度。当前 `page_size=1`，所以通常全是 1。
- `num_qo_heads`：Q/O head 数。
- `num_kv_heads`：KV head 数。
- `head_dim`：head 维度。
- `page_size`：当前强制为 1。
- `seq_lens_cpu`：每条请求的 key 长度。
- `dtype`：KV cache dtype。
- `wrapper`：本 batch 使用的 FlashInfer wrapper。
- `initialized`：是否已经调用过 wrapper.plan。

`__post_init__` 检查 page size 和 tensor device 位置，避免 CPU/GPU buffer 用错。

## `get_last_indices`

```python
return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1
```

它返回每条请求最后一个 query token 的扁平索引。

prefill 阶段 `LMHead` 会用它从所有 prompt token hidden state 中取最后 token。

## `FlashInferBackend.__init__`

初始化时创建 FlashInfer wrapper：

- `BatchPrefillWithPagedKVCacheWrapper`
- `BatchDecodeWithPagedKVCacheWrapper`

二者共享：

```python
self.float_workspace_buffer
self.int_workspace_buffer
```

共享 int workspace 可以减少重复分配。

它还保存：

- 模型配置。
- KV cache 引用。
- device。
- head 数。
- graph capture 相关结构。
- CUDA event `last_event`。

## `_initialize_metadata_once`

这个方法确保每个 `FIMetadata` 只执行一次 FlashInfer `plan()`。

它先等待上一次 event：

```python
self.last_event.synchronize()
```

原因是 FlashInfer plan 会使用 pinned host staging storage。如果前一次异步 H2D 拷贝还没结束，直接复用 CPU buffer 可能有风险。

然后根据 wrapper 类型调用不同的 `plan()`：

- prefill wrapper 使用 `qo_indptr`、`paged_kv_indptr` 等参数。
- decode wrapper 使用 `indptr`、`indices`、`last_page_len` 等参数。

最后记录新的 CUDA event。

## `_get_ones_cpu`

返回长度为 `bs` 的 int32 pinned CPU tensor，内容全是 1。

如果缓存长度不够，会按 2 的幂扩容：

```python
_next_power_of_2(bs)
```

当前 page size 为 1，所以每个请求的 last page length 都是 1。

## `prepare_metadata`

这是 scheduler 和 FlashInfer 后端之间的关键桥梁。

它从 `batch.padded_reqs` 构造：

```text
seqlens_q = req.extend_len
seqlens_k = req.device_len
cached_lens = req.cached_len
```

根据 batch 类型生成 `cu_seqlens_q_cpu`：

- decode 时 `max_seqlen_q == 1`，使用 `arange(padded_size + 1)`。
- 全量 prefill 且 cached_len 都为 0 时，Q/K cumulative length 相同。
- 其他情况按 `seqlens_q` 单独 cumsum。

然后从全局 `page_table` 取每个请求当前可见长度内的物理 KV slot：

```python
indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs])
```

最后构造 `FIMetadata` 并写到：

```python
batch.attn_metadata
```

## `forward`

核心流程：

```text
metadata 初始化 plan
-> self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
-> 取当前 layer 的完整 K/V cache
-> flatten 成 FlashInfer 需要的 paged KV cache 视图
-> metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)
```

注意 K/V cache 写入发生在 attention 计算前。这样当前 token 的 K/V 也能参与本次 attention。

## CUDA graph 相关方法

### `init_capture_graph`

创建 `FICaptureData`，并记录可 capture 的 batch size 列表。

### `use_tensor_cores`

```python
return self.config.num_qo_heads // self.config.num_kv_heads >= 4
```

当 Q head 和 KV head 比例较大时，decode wrapper 使用 tensor cores。

### `prepare_for_capture`

为某个 batch size 创建 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`。

然后调用 `prepare_metadata`，把 metadata 的 wrapper 替换成 graph wrapper，并提前初始化 plan。

### `prepare_for_replay`

replay 前把当前 batch 的 metadata wrapper 指向已经 capture 好的 graph wrapper，并初始化 metadata。

## 总结

`attention/fi.py` 是 AIOS attention 后端的核心实现。它把 scheduler 准备的 batch/page table 转成 FlashInfer metadata，在每层 attention 中写入 KV cache 并运行 paged KV attention，同时提供 decode 阶段 CUDA graph 的 capture/replay 支持。

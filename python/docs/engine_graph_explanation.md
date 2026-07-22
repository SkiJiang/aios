# `python/aios/engine/graph.py` 源码解释报告

## 文件定位

`graph.py` 实现 decode 阶段 CUDA graph capture/replay。

目标是减少 decode 小 batch 中 Python 调度和 kernel launch 开销。

## `GraphCaptureBuffer`

这个 dataclass 保存 CUDA graph 使用的静态输入输出 buffer：

- `input_ids`
- `out_loc`
- `positions`
- `logits`

CUDA graph replay 要求张量地址稳定，所以运行时不能换新 tensor，只能把新数据 copy 到这些固定 buffer。

### `init`

根据最大 batch size 创建 buffer：

```python
input_ids: [bs]
out_loc: [bs]
positions: [bs]
logits: [bs, vocab_size]
```

### `set_batch`

把 batch 的输入字段指向静态 buffer slice。

capture 时使用它固定图中的 tensor 地址。

### `copy_from`

replay 前把真实 batch 的 `input_ids/out_loc/positions` 拷贝到静态 buffer。

## `determine_cuda_graph_bs`

如果用户显式传入 `cuda_graph_bs`，直接使用。

否则根据空闲显存和最大 batch size 生成 bucket：

```text
[1, 2, 4] + [8, 16, 24, ...]
```

显存大于 80 GiB 默认最大 256，否则默认最大 160。

## `get_free_memory` 和 `mem_gb`

`get_free_memory` 调用：

```python
torch.cuda.mem_get_info(device)[0]
```

返回空闲显存字节数。

`mem_gb` 把字节数格式化为 GiB 字符串。

## `GraphRunner`

`GraphRunner` 管理多个 batch size bucket 的 CUDA graph。

初始化时确定：

- attention backend。
- device。
- dummy request。
- stream。
- graph batch size 列表。
- graph map。
- capture buffer。

然后调用 `_capture_graphs`。

## `_capture_graphs`

如果没有 graph batch size，直接返回。

否则：

1. 调用 `attn_backend.init_capture_graph`。
2. 同步并清理 CUDA memory cache。
3. 初始化 `GraphCaptureBuffer`。
4. 对每个 batch size 从大到小 capture。
5. 构造只包含 dummy request 的 decode batch。
6. 调用 `attn_backend.prepare_for_capture(batch)`。
7. 用静态 buffer 设置 batch。
8. 先 warmup 执行一次模型 forward。
9. 在 `torch.cuda.graph(...)` 中 capture 模型 forward。
10. 保存到 `graph_map[bs]`。

## `can_use_cuda_graph`

```python
return batch.is_decode and batch.size <= self.max_graph_bs
```

当前只对 decode batch 使用 CUDA graph。

prefill batch 长度变化大，不适合这个路径。

## `pad_batch`

如果 batch 可以使用 CUDA graph，就找到不小于真实 batch size 的 bucket：

```python
padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
```

然后追加 dummy request：

```python
batch.padded_reqs = batch.reqs + [dummy_req] * (...)
```

这样 batch shape 和已 capture 的 graph 匹配。

## `replay`

流程：

1. 确认 batch 可用 CUDA graph。
2. 把真实 batch 数据 copy 到静态 buffer。
3. 调用 `attn_backend.prepare_for_replay(batch)`。
4. replay 对应 padded size 的 graph。
5. 返回真实 batch size 范围内的 logits。

padding 部分 logits 被丢弃。

## `destroy_cuda_graphs`

清空 graph map 和 buffer，并调用 `gc.collect()`。

## 总结

`engine/graph.py` 通过静态 buffer、dummy request 和 batch size bucket，为 decode 阶段提供 CUDA graph replay 能力。它不改变模型语义，只优化运行时调度开销。

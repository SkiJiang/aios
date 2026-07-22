# `python/aios/scheduler/scheduler.py` 源码解释报告

## 文件定位

`scheduler.py` 是 continuous batching 调度器总入口。

它组合：

- `PrefillManager`
- `DecodeManager`
- `CacheManager`
- `TableManager`
- attention backend
- 可选 `GraphRunner`

负责把请求调度成一次次模型 forward 所需的 `Batch`。

## 类型定义

```python
Indice2D = Tuple[torch.Tensor, torch.Tensor]
```

表示二维索引：

```text
(row_indices, column_indices)
```

用于索引 `token_pool` 和 `page_table`。

## `ForwardInput`

```python
class ForwardInput(NamedTuple):
    batch: Batch
    input_tuple: Indice2D
    write_tuple: Indice2D
```

它是 scheduler 输出给 `LLM.generate()` 的对象。

- `batch`：本次 forward 的 batch。
- `input_tuple`：用于从 `token_pool` 读取本次输入 token。
- `write_tuple`：用于把采样得到的 next token 写回 `token_pool`。

## `Scheduler.__init__`

初始化时保存资源：

- table manager。
- cache manager。
- EOS token id。
- device。
- attention backend。
- prefill token budget。
- graph runner。

并创建：

```python
self.decode_manager = DecodeManager(page_size=1)
self.prefill_manager = PrefillManager(...)
self.finished = []
self._next_uid = 0
```

## `add_request`

```python
uid = self._next_uid
self._next_uid += 1
self.prefill_manager.add_one_req(PendingReq(...))
```

新请求先进入 pending 队列，不会立即分配 KV cache page。

## `schedule_next_batch`

调度策略是 prefill 优先：

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

如果能调度 prefill，就先处理新请求。

否则处理 decode batch。

拿到 batch 后调用 `_prepare_batch`。

## `_prepare_batch`

这是 scheduler 最关键的方法。

流程：

1. 如果启用 CUDA graph，调用 `graph_runner.pad_batch(batch)`。
2. 否则 `batch.padded_reqs = batch.reqs`。
3. 调用 `cache_manager.allocate_paged(batch.reqs)` 分配 KV page。
4. 生成 `batch.positions`。
5. 生成 `input_mapping`。
6. 生成 `write_mapping`。
7. 根据 page table 得到 `batch.out_loc`。
8. 调用 `attn_backend.prepare_metadata(batch)`。
9. 返回 `ForwardInput`。

其中：

```python
batch.out_loc = self.table_manager.page_table[input_mapping]
```

得到本次每个输入 token 的物理 KV cache 写入位置。

## `process_batch_output`

模型 forward 后，engine 返回 `next_tokens`。

处理流程：

1. 把 next token 写回 `token_pool[write_mapping]`。
2. 更新 decode manager 的 running set。
3. 把 next token 拷贝到 CPU。
4. 调用 `req.append_host(...)` 更新请求。
5. 判断是否完成。
6. 完成则释放 KV cache 和 table slot。
7. 加入 `finished` 列表。

## `_is_finished`

请求结束条件：

```text
命中 EOS 且没有 ignore_eos
或 req.can_decode 为 False
```

## `_free_req_resources`

释放两类资源：

```python
self.cache_manager.free_req(req)
self.table_manager.free(req.table_idx)
```

## `has_work`

```python
return self.prefill_manager.runnable or self.decode_manager.runnable
```

只要还有 pending 或 running 请求，generate 循环就继续。

## `collect_results`

把 finished 请求转成结果 dict：

```text
uid
token_ids
text
```

然后按 uid 排序，保证输出顺序和输入请求顺序一致。

## `_make_positions`

为 `batch.padded_reqs` 生成 position ids。

对每个请求生成：

```text
cached_len, cached_len + 1, ..., device_len - 1
```

这正好对应本次新增处理的 token。

## `_make_input_tuple`

构造从 `token_pool` 读取输入 token 的二维索引。

row 是 `req.table_idx`，column 是 `batch.positions`。

## `_make_write_tuple`

构造写回 next token 的二维索引。

row 是真实请求的 `table_idx`。

column 是：

```python
req.device_len if req.can_decode else -1
```

即 next token 应写入的 logical position。

## 总结

`scheduler.py` 是 AIOS continuous batching 的核心。它把 pending 和 running 请求调度成 prefill/decode batch，准备模型 forward 所需的 token、position、KV 写入位置和 attention metadata，并在 forward 后更新请求状态和释放资源。

# `python/aios/scheduler/prefill.py` 源码解释报告

## 文件定位

`prefill.py` 定义 `PrefillManager`，负责管理还没有进入模型的 pending 请求，并把可运行请求组成 prefill batch。

## `PrefillManager`

```python
@dataclass
class PrefillManager:
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq]
```

它需要同时查看：

- table slot 是否够。
- KV cache 空间是否够。
- decode 中已有请求未来还需要多少 token。

## `add_one_req`

```python
self.pending_list.append(pending)
```

把新请求加入等待队列。

`Scheduler.add_request()` 会调用这个方法。

## `runnable`

```python
return bool(self.pending_list)
```

只要 pending list 非空，就可能有 prefill 工作。

## `_can_admit`

这个方法判断一个 pending 请求是否可以进入系统。

先检查 table slot：

```python
if self.table_manager.available_size - scheduled_count <= 0:
    return False
```

再检查 KV cache 空间：

```python
needed = pending.input_len + pending.output_len
reserved = self.decode_manager.inflight_tokens + scheduled_reserved
return (needed + reserved) <= self.cache_manager.available_size
```

这里不仅考虑 prompt token，还预留最大生成 token 的空间，避免请求进入后中途没有 KV cache。

## `schedule_next_batch`

这是 prefill 调度核心。

流程：

1. 如果没有 pending 请求，返回 `None`。
2. 按顺序遍历 pending 请求。
3. 受 `prefill_budget` 限制选择请求。
4. 检查 table slot 和 KV cache 空间。
5. 选中的请求从 `pending_list` 移除。
6. 为每个请求分配 table slot。
7. 把 prompt token 拷贝到 GPU `token_pool`。
8. 创建运行时 `Req`。
9. 返回 `Batch(reqs=reqs, phase="prefill")`。

## prompt token 拷贝

```python
device_ids = self.table_manager.token_pool[table_idx, :prompt_len]
host_ids = pending.input_ids.to(torch.int32)
if not host_ids.is_pinned():
    host_ids = host_ids.pin_memory()
device_ids.copy_(host_ids, non_blocking=True)
```

这一步把 CPU prompt token 写入 GPU token pool。

后续 `Scheduler._prepare_batch()` 会生成 input mapping，`LLM.generate()` 再通过 token pool 取出当前 batch 的 `input_ids`。

## 总结

`PrefillManager` 负责 admission control 和 prefill batch 构造。它在请求进入模型前分配 table slot、预留 KV cache 空间，并把 prompt token 写入 GPU token pool。

# `python/aios/scheduler/decode.py` 源码解释报告

## 文件定位

`decode.py` 定义 `DecodeManager`，负责管理已经完成 prefill、正在逐 token 生成的运行中请求。

## `DecodeManager`

```python
@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
```

`running_reqs` 是当前仍可继续 decode 的请求集合。

`Req` dataclass 使用 `eq=False`，因此可以按对象身份放入 set。

## `filter_reqs`

```python
self.running_reqs = {
    req for req in self.running_reqs.union(reqs) if req.can_decode
}
```

这个方法把本次 batch 的请求加入 running set，并过滤掉不能继续 decode 的请求。

它通常在 `Scheduler.process_batch_output()` 中调用。

## `remove_req`

```python
self.running_reqs.discard(req)
```

当请求命中 EOS 或达到长度上限时，从运行集合中移除。

## `inflight_tokens`

```python
tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
return sum(req.remain_len for req in self.running_reqs) + tokens_reserved
```

它估算正在运行的请求未来还需要多少 KV cache token 空间。

`PrefillManager._can_admit()` 会用它判断是否还能接纳新请求。

当前 `page_size=1`，所以 `tokens_reserved` 为 0。

## `schedule_next_batch`

```python
if not self.runnable:
    return None
return Batch(reqs=list(self.running_reqs), phase="decode")
```

decode 阶段把所有运行请求组成一个 batch。

这体现 continuous batching 的典型策略：每一步 decode 尽量把所有活跃请求一起送进模型。

## `runnable`

```python
return len(self.running_reqs) > 0
```

只要还有运行请求，就可以继续 decode。

## 总结

`DecodeManager` 管理活跃请求集合，并在每个 decode step 生成一个 decode batch。它和 `PrefillManager` 配合，实现 pending、running、finished 三类请求状态流转。

# `python/aios/scheduler/common.py` 源码解释报告

## 文件定位

`scheduler/common.py` 定义 scheduler 子模块共享的小数据结构 `PendingReq`。

## `PendingReq`

```python
@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
```

它表示一个等待进入 prefill 阶段的请求。

字段含义：

- `uid`：请求编号，用于最终结果排序。
- `input_ids`：CPU 上的 prompt token。
- `sampling_params`：该请求的生成参数。

## `input_len`

```python
return len(self.input_ids)
```

返回 prompt token 数。

`PrefillManager` 会用它计算 prefill token budget 和 KV cache 需求。

## `output_len`

```python
return self.sampling_params.max_tokens
```

返回该请求最多生成多少 token。

调度器会用：

```text
input_len + output_len
```

估算请求需要保留多少 KV cache 空间。

## 总结

`PendingReq` 是请求进入系统前的轻量表示。它还没有分配 table slot，也没有创建运行时 `Req`，只保存 prompt、采样参数和 uid。

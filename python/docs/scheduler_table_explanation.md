# `python/aios/scheduler/table.py` 源码解释报告

## 文件定位

`table.py` 定义 `TableManager`，负责管理请求槽位、`page_table` 和 `token_pool`。

## `TableManager.__init__`

```python
def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
    self._free_slots = list(range(max_running_reqs))
    self.page_table = page_table
    self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```

`_free_slots` 保存可用请求行号。

`page_table` 是 GPU int32 张量，形状通常是：

```text
[max_running_reqs + 1, aligned_max_seq_len]
```

其中最后一行可能被 dummy request 使用。

`token_pool` 和 `page_table` 形状相同，用于保存每个请求每个 logical position 的 token id。

## `available_size`

```python
return len(self._free_slots)
```

返回还可以接纳多少个新请求。

`PrefillManager._can_admit()` 会用这个值限制并发请求数。

## `allocate`

```python
return self._free_slots.pop()
```

为新请求分配一个 table row。

这个 row 会写入 `Req.table_idx`，后续用于访问：

```text
page_table[table_idx, ...]
token_pool[table_idx, ...]
```

## `free`

```python
self._free_slots.append(slot)
```

请求结束后释放 table row，使后续新请求可以复用。

## 和 scheduler 的关系

prefill 时：

```text
TableManager.allocate()
-> prompt token 写入 token_pool[table_idx, :prompt_len]
```

forward 前：

```text
batch.input_ids = token_pool[input_tuple]
batch.out_loc = page_table[input_tuple]
```

生成后：

```text
token_pool[write_tuple] = next_tokens
```

请求结束：

```text
TableManager.free(req.table_idx)
```

## 总结

`TableManager` 管理请求在 GPU 表中的行号，并维护 token id 的设备侧存储。它让 scheduler 可以用二维索引把请求 logical position 映射到当前 batch 的输入 token。

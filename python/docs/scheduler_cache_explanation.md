# `python/aios/scheduler/cache.py` 源码解释报告

## 文件定位

`scheduler/cache.py` 定义 scheduler 使用的物理 KV page 分配器 `CacheManager`。

它不保存 K/V tensor 本身，而是管理 `page_table` 中 logical position 到物理 cache slot 的映射。

## `_div_ceil`

```python
def _div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b
```

向上整除，用于根据 token 长度计算需要多少 page。

## `CacheManager.__init__`

```python
self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
```

`free_slots` 保存空闲物理 page 起始 token slot。

如果 `page_size=1`，它就是：

```text
0, 1, 2, ..., num_pages - 1
```

还保存：

- `device`
- `num_pages`
- `page_table`
- `page_size`

## `available_size`

```python
return len(self.free_slots) * self.page_size
```

返回还可容纳多少 token。

`PrefillManager` 会用它判断新请求能否进入。

## `allocate_paged`

这是核心方法。

对每个请求：

```python
first_page = _div_ceil(req.cached_len, self.page_size)
last_page = _div_ceil(req.device_len, self.page_size)
```

`cached_len` 之前的 token 已经有 cache page。

`device_len` 表示本次 forward 需要覆盖到的位置。

如果 `last_page > first_page`，说明需要为新增 token 分配 page。

分配后调用 `_write_page_table`：

```python
page_table[table_idx, logical_position] = physical_slot
```

## `free_req`

```python
indices = self.page_table[req.table_idx, : req.cached_len]
self._free(indices)
```

请求结束时，根据已经缓存的 token 释放对应物理位置。

## `_allocate`

从 `free_slots` 前部切出需要的 page。

如果空闲 page 不够，抛：

```python
RuntimeError("KV cache exhausted: ...")
```

## `_free`

把释放的 page 追加回 `free_slots`。

当前实现不排序，也不合并，因为 page size 目前为 1，逻辑较简单。

## `_page_to_token`

如果 `page_size=1`，page id 就是 token slot。

如果 page size 大于 1，会展开成 page 内所有 token slot。

## `_write_page_table`

这个函数用 pinned CPU tensor 构造二维索引：

- `table_idx_host`
- `positions_host`

再异步拷贝到 GPU：

```python
table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
positions = positions_host.to(page_table.device, non_blocking=True)
```

最后写：

```python
page_table[table_idxs, positions] = allocated
```

## 总结

`scheduler/cache.py` 是 KV cache 物理页分配器。它把请求的 logical token position 映射到全局物理 cache slot，为后续 `batch.out_loc` 和 `store_cache` 写入提供位置依据。

# `python/aios/kvcache/naive_manager.py` 源码解释报告

## 文件定位

`naive_manager.py` 提供一个最简单的 cache manager 实现，主要用于满足 `BaseCacheManager` 接口。

它不实现真正的 prefix cache、锁定、驱逐等复杂逻辑。

## `NaiveCacheHandle`

```python
class NaiveCacheHandle(BaseCacheHandle):
    pass
```

它没有新增字段，只继承 `BaseCacheHandle` 的 `cached_len`。

## `NaiveCacheManager`

初始化：

```python
self.device = device
self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
```

`empty_tensor` 用于表示没有命中的缓存位置。

## `match_prefix`

```python
return NaiveCacheHandle(0), self.empty_tensor
```

无论输入是什么，都返回缓存长度 0，表示没有 prefix cache 命中。

## `lock_handle`

这个方法什么也不做：

```python
_ = handle, unlock
```

因为 naive 实现没有真实 handle 生命周期管理。

## `insert_prefix`

```python
assert len(indices) == len(input_ids)
return len(indices)
```

它只检查 token 数和 indices 数一致，然后返回插入长度。

没有真正保存 prefix cache。

## `evict`

如果驱逐大小是 0，返回空 tensor。

如果 size 大于 0，抛：

```python
NotImplementedError("NaiveCacheManager does not support eviction.")
```

## `reset`、`size_info`、`check_integrity`

`reset` 和 `check_integrity` 都是空操作。

`size_info` 返回：

```python
SizeInfo(evictable_size=0, protected_size=0)
```

表示没有可驱逐或受保护缓存。

## 总结

`NaiveCacheManager` 是一个不做 prefix cache 的占位实现。它满足抽象接口，但所有请求都视为没有可复用 prefix，适合简单路径或后续 prefix cache 功能接入前的默认实现。

# `python/aios/kvcache/base.py` 源码解释报告

## 文件定位

`kvcache/base.py` 定义 KV cache 和 cache manager 的抽象接口。

它不提供具体存储实现，而是规定后续实现必须支持哪些操作。

## `BaseKVCache`

`BaseKVCache` 是真实 KV cache 存储的抽象基类。

必须实现：

- `k_cache(index)`：返回某一层的 K cache。
- `v_cache(index)`：返回某一层的 V cache。
- `store_kv(k, v, out_loc, layer_id)`：把当前 token 的 K/V 写入 cache。
- `device`：cache 所在设备。
- `dtype`：cache dtype。
- `num_layers`：层数。

`FlashInferBackend.forward()` 依赖这些接口。

## `BaseCacheHandle`

```python
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int
```

它表示 prefix cache 命中结果或缓存句柄。

当前简单实现中只记录 `cached_len`。

## `SizeInfo`

```python
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int
```

它描述 cache manager 中可驱逐和受保护的缓存大小。

`total_size` 返回二者之和。

## `BaseCacheManager`

这是 prefix cache 管理策略的抽象接口。

必须实现：

- `match_prefix(input_ids)`：匹配已有 prefix cache。
- `lock_handle(handle, unlock=False)`：锁定或解锁 cache handle。
- `insert_prefix(input_ids, indices)`：插入新的 prefix cache。
- `evict(size)`：驱逐指定大小缓存。
- `reset()`：重置缓存管理器。
- `size_info`：返回缓存大小信息。
- `check_integrity()`：检查内部一致性。

当前主调度路径使用 `scheduler/cache.py` 进行物理 page 分配；这里的 `BaseCacheManager` 更偏向 prefix cache 能力预留。

## 总结

`kvcache/base.py` 定义了 KV cache 存储与 prefix cache 管理的接口边界。真实 KV 张量存储由 `MHAKVCache` 实现，简单 prefix cache 管理由 `NaiveCacheManager` 实现。

# `python/aios/kernel/__init__.py` 源码解释报告

## 文件定位

`kernel/__init__.py` 是 `aios.kernel` 包的导出入口。

当前只导出一个函数：

```python
from .store import store_cache
```

## `store_cache`

`store_cache` 是 Triton kernel 的 Python 包装函数，用于把当前 batch 的 K/V 写入 KV cache。

它在 `MHAKVCache.store_kv()` 中被调用：

```python
from aios.kernel import store_cache
```

## `__all__`

```python
__all__ = ["store_cache"]
```

这说明当前 kernel 子包对外只暴露 KV cache 写入能力。

## 总结

`kernel/__init__.py` 是很小的门面文件，把底层 Triton 写 cache 函数暴露给 KV cache 模块使用。

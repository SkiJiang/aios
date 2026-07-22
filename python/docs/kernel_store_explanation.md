# `python/aios/kernel/store.py` 源码解释报告

## 文件定位

`kernel/store.py` 定义一个 Triton kernel，用于把当前 batch 计算出的 K/V tensor scatter 写入连续 KV cache。

它是 attention backend 和 KV cache 存储之间的底层写入路径。

## `_store_cache_kernel`

```python
@triton.jit
def _store_cache_kernel(...)
```

这是实际运行在 GPU 上的 Triton JIT kernel。

每个 program 处理一个 token：

```python
token_idx = tl.program_id(axis=0)
```

`indices[token_idx]` 给出这个 token 应写入的物理 cache slot：

```python
index = tl.load(indices_ptr + token_idx).to(tl.int64)
```

## 扁平宽度

`width` 表示一个 token 的 K 或 V 向量展平后的元素数：

```text
num_heads * head_dim
```

kernel 用：

```python
offsets = tl.arange(0, block_size)
mask = offsets < width
```

生成向量化 load/store 的 offset。

## 读取 K/V

```python
k_values = tl.load(k_ptr + token_idx * k_token_stride + offsets, mask=mask)
v_values = tl.load(v_ptr + token_idx * v_token_stride + offsets, mask=mask)
```

从输入 batch 的第 `token_idx` 个 token 读取 K/V。

## 写入 cache

```python
cache_offsets = index * cache_token_stride + offsets
tl.store(k_cache_ptr + cache_offsets, k_values, mask=mask)
tl.store(v_cache_ptr + cache_offsets, v_values, mask=mask)
```

把 K/V 写到物理 slot `index` 对应的位置。

## `store_cache`

这是 Python 包装函数，负责输入检查和 launch kernel。

检查包括：

- `k_cache/v_cache` 必须是三维。
- `k/v` 的 head 和 dim 必须匹配 cache 布局。
- `indices` 数量必须等于 token 数。
- `k` 和 `v` 形状必须一致。
- 所有输入必须在 CUDA 上。

然后计算：

```python
width = k_cache.shape[1] * k_cache.shape[2]
block_size = triton.next_power_of_2(width)
```

最后以 token 数作为 grid：

```python
_store_cache_kernel[(k.shape[0],)](...)
```

## 调用链

典型调用链：

```text
Qwen3Attention.forward
-> FlashInferBackend.forward
-> MHAKVCache.store_kv
-> store_cache
-> _store_cache_kernel
```

## 总结

`kernel/store.py` 是 KV cache 写入的底层实现。它用 Triton 按 token 并行，把扁平 batch 的 K/V 根据 `indices` scatter 到物理 KV cache slot 中。

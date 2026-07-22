# `python/aios/kvcache/mha_pool.py` 源码解释报告

## 文件定位

`mha_pool.py` 实现真实的多头 attention KV cache 存储类 `MHAKVCache`。

它被 `Engine` 创建，并挂到全局 `Context` 中，供 `FlashInferBackend` 使用。

## 初始化参数

`MHAKVCache.__init__` 接收：

- `num_kv_heads`：KV head 数。
- `num_layers`：模型层数。
- `head_dim`：每个 head 的维度。
- `num_pages`：KV cache page 数。
- `page_size`：每页 token 数。当前必须为 1。
- `dtype`：cache dtype。
- `device`：CUDA 设备。

断言：

```python
assert device.type == "cuda"
assert page_size == 1
```

当前实现只支持 CUDA 和 FlashInfer page size 1。

## KV buffer 形状

```python
self._kv_buffer = torch.empty(
    (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
    device=device,
    dtype=dtype,
)
```

维度含义：

```text
2              K/V 两类 cache
num_layers     decoder 层数
num_pages      物理 page 数
page_size      每页 token 数
local_kv_heads KV head 数
head_dim       head 维度
```

随后切出：

```python
self._k_buffer = self._kv_buffer[0]
self._v_buffer = self._kv_buffer[1]
```

## `k_cache` 和 `v_cache`

```python
def k_cache(self, index: int) -> torch.Tensor:
    return self._k_buffer[index]
```

返回指定 layer 的 K cache。

`v_cache` 同理返回指定 layer 的 V cache。

## `store_kv`

```python
def store_kv(self, k, v, out_loc, layer_id) -> None:
    from aios.kernel import store_cache
```

它把当前 batch 的 K/V 写入指定 layer 的 KV cache。

调用时会把 layer cache view 成：

```text
(num_pages * page_size, local_kv_heads, head_dim)
```

然后交给 `kernel/store.py` 中的 Triton kernel scatter 写入。

`out_loc` 是每个 token 应写入的物理 slot，由 scheduler 根据 page table 生成。

## 属性

- `device`：返回 cache 所在设备。
- `dtype`：返回 `_kv_buffer.dtype`。
- `num_layers`：返回模型层数。

这些属性用于 attention backend 和 engine 查询 cache 信息。

## 总结

`MHAKVCache` 是 AIOS 当前真实的 KV cache 存储池。它按 K/V、layer、page、head、dim 组织显存，并通过 Triton `store_cache` kernel 把每次 forward 产生的 K/V 写入对应物理位置。

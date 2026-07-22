# `python/aios/attention/base.py` 源码解释报告

## 文件定位

`attention/base.py` 定义 attention backend 的抽象接口。

模型层 `Qwen3Attention` 不直接实现 attention 计算，而是调用：

```python
ctx.attn_backend.forward(...)
```

因此所有 attention backend 都必须遵守这里的接口。

## `BaseAttnMetadata`

```python
@dataclass
class BaseAttnMetadata(ABC):
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...
```

metadata 负责保存一次 batch attention 需要的辅助信息。

唯一抽象方法 `get_last_indices(bs)` 用于返回每条请求最后一个 query token 的索引。

这个方法被 `LMHead.forward()` 使用：

```python
indices = batch.attn_metadata.get_last_indices(batch.size)
x = x[indices].contiguous()
```

prefill 阶段 prompt 有多个 token，但采样时只需要最后 token 的 logits，所以需要这个接口。

## `BaseAttnBackend`

这个抽象类定义 attention backend 必须实现的方法。

### `forward`

```python
def forward(self, q, k, v, layer_id, batch) -> torch.Tensor
```

执行一次 attention。

输入：

- `q/k/v`：当前 layer 计算出的 query/key/value。
- `layer_id`：当前 decoder layer 编号。
- `batch`：当前 forward batch。

后端通常会在这里把 K/V 写入 KV cache，然后调用具体 attention kernel。

### `prepare_metadata`

```python
def prepare_metadata(self, batch: Batch) -> None
```

在模型 forward 前，根据 batch 的请求长度、page table、phase 等信息准备 metadata。

Scheduler 会在 `_prepare_batch` 中调用它。

### CUDA graph 相关接口

```python
init_capture_graph(max_seq_len, bs_list)
prepare_for_capture(batch)
prepare_for_replay(batch)
```

这些接口用于 decode 阶段 CUDA graph capture/replay。

不同 attention backend 需要不同静态 buffer 和 wrapper，因此 graph 准备逻辑放在 backend 内。

## `HybridBackend`

`HybridBackend` 接收两个 backend：

```python
prefill_backend
decode_backend
```

它根据 batch phase 分发：

```python
backend = self.prefill_backend if batch.is_prefill else self.decode_backend
```

`forward` 和 `prepare_metadata` 都按 phase 选择对应 backend。

CUDA graph 相关方法只委托给 decode backend，因为 CUDA graph 通常用于 shape 更稳定的 decode 阶段。

## 总结

`attention/base.py` 定义了 AIOS attention 后端的统一契约。模型只依赖这个接口，不关心实际使用 FlashInfer、普通 PyTorch 还是其他 backend，从而保持模型结构和底层 attention 实现解耦。

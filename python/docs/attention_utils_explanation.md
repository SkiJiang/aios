# `python/aios/attention/utils.py` 源码解释报告

## 文件定位

`attention/utils.py` 定义 attention backend 在 CUDA graph capture/replay 中使用的基础数据容器 `BaseCaptureData`。

它没有执行 attention 计算，只负责创建静态 buffer。

## `BaseCaptureData`

```python
@dataclass
class BaseCaptureData:
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
```

这些字段都是 CUDA graph 场景下需要固定地址的张量。

字段含义：

- `seq_lens`：每条请求的序列长度。
- `positions`：当前 token 的 position id。
- `cu_seqlens_k`：K 序列长度的 cumulative sum。
- `cu_seqlens_q`：Q 序列长度的 cumulative sum。
- `page_table`：batch 对应的 page table 或展开后的 indices buffer。

## `create`

```python
@classmethod
def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
```

这个类方法创建一组默认张量：

```python
seq_lens=torch.ones(max_bs, dtype=torch.int32, device=device)
positions=torch.zeros(max_bs, dtype=torch.int32, device=device)
cu_seqlens_k=torch.arange(max_bs + 1, dtype=torch.int32, device=device)
cu_seqlens_q=torch.arange(max_bs + 1, dtype=torch.int32, device=device)
page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device)
```

`**kwargs` 允许子类补充或覆盖字段。

## 和 FlashInfer 的关系

`attention/fi.py` 中的 `FICaptureData` 继承 `BaseCaptureData`，并把：

```python
one_tensor -> seq_lens
indices -> page_table
```

作为 FlashInfer CUDAGraph wrapper 需要的 buffer。

## 总结

`attention/utils.py` 为 attention backend 的 CUDA graph 支持提供基础静态张量容器。它的核心价值是保证 capture 和 replay 使用同一组固定地址的输入 buffer。

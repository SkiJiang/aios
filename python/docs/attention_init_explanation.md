# `python/aios/attention/__init__.py` 源码解释报告

## 文件定位

这个文件是 `aios.attention` 包的统一导出入口。它把 attention backend 的抽象类、FlashInfer 实现和兼容别名集中暴露给上层。

## 导出内容

```python
from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend
from .fi import FICaptureData, FIMetadata, FlashInferBackend
```

主要对象：

- `BaseAttnBackend`：attention 后端抽象接口。
- `BaseAttnMetadata`：attention metadata 抽象接口。
- `HybridBackend`：prefill/decode 分发后端。
- `FICaptureData`：FlashInfer CUDA graph capture 数据。
- `FIMetadata`：FlashInfer 单次 batch metadata。
- `FlashInferBackend`：FlashInfer attention 后端实现。

## 兼容别名

源码中还定义了几组旧命名别名：

```python
BaseAttentionBackend = BaseAttnBackend
BaseAttentionMetadata = BaseAttnMetadata
HybridAttentionBackend = HybridBackend
FlashInferAttentionBackend = FlashInferBackend
FlashInferAttentionMetadata = FIMetadata
```

这些别名用于兼容早期代码。新代码应使用短命名，例如 `BaseAttnBackend`。

## `__all__`

`__all__` 只列出了新命名对象，没有列出兼容别名。

这表示包的正式公共 API 是：

```text
BaseAttnBackend
BaseAttnMetadata
HybridBackend
FICaptureData
FIMetadata
FlashInferBackend
```

## 总结

`attention/__init__.py` 是 attention 子包的门面。它统一导出后端抽象和 FlashInfer 实现，同时保留旧名称兼容，但正式 API 以短命名为准。

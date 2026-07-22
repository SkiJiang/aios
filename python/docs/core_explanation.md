# `python/aios/core.py` 源码解释报告

## 文件定位

`core.py` 定义 AIOS 推理运行时的核心数据结构。它被 `llm`、`engine`、`scheduler`、`models`、`attention` 多个模块共享。

主要内容：

- `SamplingParams`：采样参数。
- `Req`：单个请求的运行时状态。
- `Batch`：一次 forward 的请求集合。
- `Context`：模型 forward 期间的全局上下文。
- `set_global_ctx / clear_global_ctx / get_global_ctx`：全局上下文管理函数。

## `SamplingParams`

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
```

它描述每个请求的生成策略。

字段含义：

- `temperature`：采样温度，`0.0` 通常表示贪心。
- `top_k`：只在概率最高的 k 个 token 中采样，`-1` 表示关闭。
- `top_p`：nucleus sampling 参数。
- `ignore_eos`：是否忽略 EOS token。
- `max_tokens`：最多生成 token 数。

`is_greedy` 属性：

```python
return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

它被 `engine/sample.py` 使用。若为贪心，直接对 logits 做 `argmax`。

## `Req`

`Req` 表示一个已经进入调度系统的请求。

```python
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    generated: List[int] = field(default_factory=list)
```

重要字段：

- `input_ids`：CPU tensor，保存完整 token 序列。
- `table_idx`：该请求在 `page_table` 和 `token_pool` 中占用的行号。
- `cached_len`：已经写入 KV cache 的 token 数。
- `output_len`：最多生成多少 token。
- `uid`：请求编号，用于最终结果排序。
- `sampling_params`：该请求的采样参数。
- `generated`：已经生成的 token id 列表。

### `__post_init__`

```python
self.device_len = len(self.input_ids)
self.max_device_len = len(self.input_ids) + self.output_len
```

`device_len` 表示当前设备侧应该可见的序列长度。

`max_device_len` 表示 prompt 长度加最大生成长度。

断言：

```python
0 <= cached_len < device_len <= max_device_len
```

确保请求长度状态合法。

### 长度属性

`remain_len`：

```python
max_device_len - device_len
```

表示还可以继续生成多少 token。

`extend_len`：

```python
device_len - cached_len
```

表示本次 forward 需要新增处理的 token 数。

prefill 阶段通常 `extend_len = prompt_len`。

decode 阶段通常 `extend_len = 1`。

### `complete_one`

```python
self.cached_len = self.device_len
self.device_len += 1
```

一次模型 forward 完成后，当前可见 token 已经写入 KV cache，于是 `cached_len` 追上 `device_len`。随后 `device_len += 1`，为下一次 decode 预留一个新 token 位置。

### `append_host`

```python
self.input_ids = torch.cat([self.input_ids, next_token])
self.generated.append(int(next_token.item()))
```

采样得到的 next token 会追加到 CPU 侧 `input_ids`，同时记录到 `generated`。

## `Batch`

`Batch` 表示一次模型 forward 的请求集合。

```python
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
```

`phase` 决定这是 prefill batch 还是 decode batch。

调度器会补充这些字段：

- `input_ids`：本次 forward 的设备侧输入 token。
- `positions`：本次 forward 的 position ids。
- `out_loc`：本次 K/V 应写入的物理 cache 位置。
- `padded_reqs`：CUDA graph 场景下补齐后的请求列表。

attention backend 会补充：

- `attn_metadata`：FlashInfer 等 backend 需要的 metadata。

常用属性：

- `is_prefill`
- `is_decode`
- `size`
- `padded_size`

`size` 是真实请求数，`padded_size` 是包含 dummy request 后的 batch size。

## `Context`

`Context` 是模型 forward 使用的全局运行时上下文。

```python
@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor
    attn_backend: BaseAttnBackend
    kv_cache: MHAKVCache
    _batch: Batch | None = None
```

模型代码不显式传入 batch、KV cache 或 attention backend，而是通过：

```python
get_global_ctx()
```

访问当前上下文。

### `batch` 属性

```python
assert self._batch is not None, "No active batch in context"
return self._batch
```

只有在 `forward_batch` 上下文中才能访问当前 batch。

### `forward_batch`

```python
@contextmanager
def forward_batch(self, batch: Batch):
    assert self._batch is None, "Nested forward_batch is not allowed"
    try:
        self._batch = batch
        yield
    finally:
        self._batch = None
```

它在模型 forward 期间临时设置 active batch，结束后清空，避免 batch 状态泄漏。

`Engine.forward_batch()` 会使用：

```python
with self.ctx.forward_batch(batch):
    logits = self.model.forward()
```

## 全局上下文函数

```python
_GLOBAL_CTX: Context | None = None
```

`set_global_ctx(ctx)` 只允许设置一次，防止重复初始化。

`clear_global_ctx()` 清空上下文，通常在 `Engine.shutdown()` 中调用。

`get_global_ctx()` 要求上下文已经设置，否则抛断言错误。

## 跨模块使用关系

- `Engine.__init__` 创建并设置全局 `Context`。
- `Qwen3Model.forward` 从 `ctx.batch.input_ids` 读取输入。
- `Qwen3Attention.forward` 调用 `ctx.attn_backend.forward`。
- `LMHead.forward` 根据 `ctx.batch.is_prefill` 只取最后 token。
- `FlashInferBackend` 从 `get_global_ctx()` 获取 KV cache 和 page table。

## 总结

`core.py` 是 AIOS 的运行时语义中心。`Req` 描述单请求状态，`Batch` 描述一次 forward，`Context` 把模型结构和运行时资源连接起来。理解这个文件后，才能顺利理解 scheduler、engine、attention backend 和模型 forward 的协作方式。

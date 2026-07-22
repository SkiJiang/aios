# `python/aios/llm/llm.py` 源码解释报告

## 文件定位

`llm.py` 定义用户侧主 API `LLM`。

它把 tokenizer、模型配置、engine、scheduler、cache manager 串起来，提供 `generate()` 方法。

## `_resolve_model_path`

```python
def _resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)
```

如果传入本地目录，直接使用。

否则调用 HuggingFace Hub 的 `snapshot_download` 下载模型。

## `LLM.__init__`

初始化参数：

```python
def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs)
```

### device 和基础配置

```python
self.device = _normalize_cuda_device(kwargs.get("device", "cuda"))
assert self.device.type == "cuda"
self.dtype = dtype
self.max_running_reqs = int(kwargs.get("max_running_reqs", 16))
```

AIOS 当前只支持 CUDA。

### CUDA graph 开关

```python
self.enable_cuda_graph = bool(
    kwargs.get("enable_cuda_graph", kwargs.get("cuda_graph", False))
)
```

兼容 `enable_cuda_graph` 和 `cuda_graph` 两个参数名。

### 模型配置和 tokenizer

```python
model_path = _resolve_model_path(model_path)
hf_config = AutoConfig.from_pretrained(model_path)
config = ModelConfig.from_hf(hf_config)
self.tokenizer = AutoTokenizer.from_pretrained(model_path)
```

模型结构配置来自 HuggingFace config，再转成 AIOS 内部 `ModelConfig`。

### CUDA graph batch size 参数

`cuda_graph_bs` 支持字符串：

```text
"1,2,4,8"
```

会被解析成整数列表。

随后过滤掉大于 `max_running_reqs` 的 batch size。

`cuda_graph_max_bs` 也会被限制在 `max_running_reqs` 内。

### 创建 Engine

```python
self.engine = Engine(...)
```

Engine 会创建模型、加载权重、分配 KV cache、创建 attention backend。

LLM 还把部分 engine 字段暴露到自身：

```python
self.stream
self.model
self.attn_backend
self.page_table
self.max_seq_len
self.graph_runner
```

### 创建 scheduler CacheManager

```python
self.cache_manager = CacheManager(
    self.engine.num_pages, self.engine.ctx.page_size, self.page_table
)
```

注意这里的 `CacheManager` 是 scheduler 使用的物理 page 分配器，不是真实 K/V tensor 存储。

真实 K/V tensor 在 `Engine.kv_cache` 中。

## `close`

```python
self.engine.shutdown()
self.graph_runner = None
```

释放 graph runner 并清理全局 context。

## `generate`

这是用户生成入口，使用 `@torch.no_grad()` 禁用梯度。

### 采样参数归一化

如果没有传入，创建默认 `SamplingParams`。

如果传入单个 `SamplingParams`，复制成和 prompt 数一样的列表。

### prompt 编码

如果 prompt 是字符串：

1. 包装成 chat message。
2. 调用 tokenizer 的 `apply_chat_template`。
3. 编码成 token ids。

如果 prompt 已经是 token id list，直接转成 tensor。

### 并发和长度检查

`max_running_reqs` 被限制在：

```text
1 到 min(len(prompts), self.max_running_reqs)
```

然后检查：

```text
max(prompt_len + max_tokens) <= self.max_seq_len
```

超过则抛 `ValueError`。

### 创建 TableManager 和 Scheduler

每次 generate 前：

```python
self.engine.reset_page_table()
table_manager = TableManager(max_running_reqs, self.page_table)
```

然后创建 `Scheduler`，传入 table/cache/attention/graph 等资源。

### 主循环

```python
while scheduler.has_work:
    forward_input = scheduler.schedule_next_batch()
    batch = forward_input.batch
    batch.input_ids = table_manager.token_pool[forward_input.input_tuple]
    next_tokens = self.engine.forward_batch(batch)
    scheduler.process_batch_output(forward_input, next_tokens)
```

这个循环串起了：

```text
scheduler 准备 batch
-> 从 token_pool 取 input_ids
-> engine forward 并采样
-> scheduler 更新请求状态
```

### 返回结果

```python
return scheduler.collect_results(self.tokenizer)
```

结果会按 uid 排序，保持输入顺序。

## `_normalize_cuda_device`

如果传入 `"cuda"`，规范化为：

```python
torch.device("cuda:0")
```

如果传入 `"cuda:1"`，保留指定 index。

## 总结

`llm.py` 是 AIOS 用户 API 的核心。它负责模型路径解析、tokenizer 和 engine 初始化，并在 `generate()` 中驱动 scheduler 与 engine 完成 continuous batching 推理。

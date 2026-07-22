# `python/aios/models/config.py` 源码解释报告

## 文件定位

`config.py` 定义 `ModelConfig`，用于把 HuggingFace 模型配置或本地 `config.json` 转成 AIOS 内部需要的统一配置结构。

## `ModelConfig`

```python
@dataclass(frozen=True)
class ModelConfig:
```

`frozen=True` 表示配置对象创建后不可修改。这适合模型结构配置，因为模型初始化后这些字段不应改变。

字段包括：

- `num_layers`：decoder layer 数。
- `num_qo_heads`：query/output head 数。
- `num_kv_heads`：key/value head 数。
- `head_dim`：每个 head 的维度。
- `hidden_size`：隐藏层维度。
- `vocab_size`：词表大小。
- `intermediate_size`：MLP 中间层维度。
- `hidden_act`：MLP 激活函数名。
- `rms_norm_eps`：RMSNorm epsilon。
- `rope_theta`：RoPE 基数。
- `max_position_embeddings`：最大位置长度。
- `tie_word_embeddings`：是否共享 embedding 和 lm head 权重。

## `from_hf`

```python
@classmethod
def from_hf(cls, config) -> ModelConfig:
```

这个方法接收 HuggingFace `AutoConfig` 对象，提取 AIOS 需要的字段。

兼容逻辑：

```python
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
```

如果模型配置没有 `num_key_value_heads`，就退化为普通 MHA：KV head 数等于 attention head 数。

如果没有显式 `head_dim`，就用：

```text
hidden_size / num_attention_heads
```

计算。

其他字段也使用 `getattr` 设置默认值，例如：

- `hidden_act` 默认 `"silu"`。
- `rope_theta` 默认 `1000000.0`。
- `tie_word_embeddings` 默认 `False`。

## `from_json`

```python
@classmethod
def from_json(cls, model_path: str) -> ModelConfig:
```

这个方法直接读取：

```text
model_path/config.json
```

然后从 JSON 字典中取出模型配置。

它和 `from_hf` 的目的相同，区别是输入来源不同：

- `from_hf` 面向 transformers config object。
- `from_json` 面向本地 JSON 文件。

## 在运行时的位置

`LLM.__init__` 中会执行：

```python
hf_config = AutoConfig.from_pretrained(model_path)
config = ModelConfig.from_hf(hf_config)
```

随后 `Engine`、`Qwen3ForCausalLM`、`FlashInferBackend` 都使用这个 `ModelConfig`。

## 总结

`models/config.py` 是模型配置适配层。它把 HuggingFace 或 JSON 配置压缩成 AIOS 需要的固定字段集合，让模型结构、attention backend 和 KV cache 初始化可以使用统一配置对象。

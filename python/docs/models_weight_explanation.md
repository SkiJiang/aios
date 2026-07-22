# `python/aios/models/weight.py` 源码解释报告

## 文件定位

`weight.py` 负责从 HuggingFace safetensors checkpoint 加载权重，并适配 AIOS 模型中的 fused projection 布局。

它解决两个问题：

1. checkpoint 可能分散在多个 `.safetensors` 文件中。
2. AIOS 模型使用 `qkv_proj`、`gate_up_proj` 等合并权重，而 HF checkpoint 通常保存拆分权重。

## `packed_modules_mapping`

```python
packed_modules_mapping = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
```

这个映射定义 fused 目标权重和 HF 源权重的关系。

例如目标 key：

```text
model.layers.0.self_attn.qkv_proj.weight
```

会对应源 key：

```text
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
```

读取后沿 dim 0 拼接。

## `_checkpoint_index`

```python
def _checkpoint_index(files: Iterable[str]) -> dict[str, str]:
```

它遍历所有 safetensors 文件，建立：

```text
tensor_name -> file_path
```

索引。

如果不同文件中出现重复 key，会抛：

```python
RuntimeError("Duplicate safetensors key: ...")
```

这样后续读取某个 tensor 时无需每次扫描全部文件。

## `_read_tensor`

```python
def _read_tensor(index: dict[str, str], name: str) -> torch.Tensor:
```

它根据索引找到 tensor 所在文件，然后用 `safetensors.safe_open` 读取。

如果缺少 key，会抛更明确的错误：

```python
KeyError("Checkpoint is missing required tensor: ...")
```

## `_packed_source_names`

```python
def _packed_source_names(target_name: str) -> tuple[str, ...] | None:
```

它检查目标权重名中是否包含：

```text
.qkv_proj.
.gate_up_proj.
```

如果包含，就替换成源模块名列表。

如果不是 fused 权重，返回 `None`。

## `load_weights`

```python
def load_weights(model: BaseOP, model_path: str, device: torch.device, dtype: torch.dtype) -> None:
```

这是对外主函数。

流程：

1. 找到模型目录下所有 `.safetensors` 文件。
2. 建立 checkpoint key 到文件路径的索引。
3. 遍历 `model.state_dict()` 中的目标权重 key。
4. 判断目标 key 是否需要 fused 拼接。
5. 读取 tensor，必要时 `torch.cat(..., dim=0)`。
6. 转到目标 `device` 和 `dtype`。
7. 调用 `model.load_state_dict(fused_state_dict)` 写回模型。

## 和 `BaseOP` 的关系

这个加载器不手写模型需要哪些权重，而是依赖：

```python
for target_name in model.state_dict():
```

`BaseOP.state_dict()` 会按对象结构递归生成 key。

因此模型字段名、layer 字段名和 checkpoint key 必须保持一致。

## 设计边界

fused checkpoint 适配逻辑放在 `weight.py`，而不是放在 `LinearQKVMerged` 或 `Qwen3Attention` 里。

这样模型 forward 只关心计算，权重加载器负责外部 checkpoint 格式差异。

## 总结

`models/weight.py` 是 checkpoint 适配层。它把 HuggingFace safetensors 中的拆分权重读取、拼接、搬运到 GPU，并通过 `BaseOP.load_state_dict()` 注入 AIOS 模型。

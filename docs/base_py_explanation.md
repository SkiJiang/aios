# `python/aios/layers/base.py` 源码讲解

这个文件定义了 AIOS 里一套很轻量的“算子/层”基类体系，作用类似 PyTorch 的 `nn.Module` 加 `state_dict/load_state_dict`，但实现更简单、更可控。

它主要提供四类能力：

1. `_concat_prefix`：拼接权重名。
2. `BaseOP`：所有有状态算子的基类，负责递归收集和加载权重。
3. `StateLessOP`：无可学习权重算子的基类。
4. `OPList`：类似 `nn.ModuleList` 的容器，用来管理一组 `BaseOP` 子模块。

它服务于项目里的模型层，比如 `Linear`、`Embedding`、`RMSNorm`、`Qwen3Attention`、`Qwen3Model` 等。

## 类型定义

```python
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]
```

这表示本项目里的 `state_dict` 类型是：

```python
dict[str, torch.Tensor]
```

也就是说，权重字典的 key 是字符串，value 必须是 `torch.Tensor`。

这和 PyTorch 的 `nn.Module.state_dict()` 很像，但这里没有 `Parameter`、buffer、module registry 等复杂机制，完全依赖对象的 `__dict__`。

## 命名约定

文件开头的注释很关键：

```python
### weights: 权重成员变量，必须是 torch.Tensor 类型，且不以 _ 开头命名
### _others: 其他成员变量，必须以 _ 开头命名，且不包含在 state_dict 中
```

这个系统靠字段命名区分“权重”和“非权重”：

```python
self.weight = torch.empty(...)
self.qkv_proj = Linear(...)
self._scale = ...
self._layer_idx = ...
```

规则是：

- 不以下划线 `_` 开头的 `torch.Tensor` 会被当成权重。
- 不以下划线 `_` 开头的 `BaseOP` 会被递归遍历。
- 以下划线 `_` 开头的字段会被跳过。
- 其他公开字段，比如 `int`、`float`、`tuple`，目前会被静默忽略。

例如 `Qwen3Attention` 里：

```python
self.qkv_proj = LinearQKVMerged(...)
self.o_proj = Linear(...)
self.q_norm = RMSNorm(...)
self.k_norm = RMSNorm(...)
self._scale = ...
self._layer_idx = ...
```

`qkv_proj/o_proj/q_norm/k_norm` 会进入递归权重系统，`_scale/_layer_idx` 不会。

## `_concat_prefix`

```python
def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name
```

它用来生成类似 PyTorch 的层级权重名。

例如：

```python
_concat_prefix("", "weight")
# "weight"

_concat_prefix("model.embed_tokens", "weight")
# "model.embed_tokens.weight"
```

递归收集权重时，每深入一层，就把当前字段名拼到 prefix 后面。

## `BaseOP.forward`

```python
class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
```

设计意图是：所有继承 `BaseOP` 的算子都应该实现 `forward()`。

不过这里有一个细节：`BaseOP` 没有继承 `ABC`，所以单独这个类本身并不会像标准抽象基类那样阻止实例化。`@abstractmethod` 更多是表达接口约束。项目里的 `BaseLLMModel` 则同时继承了 `ABC` 和 `BaseOP`。

## `BaseOP.state_dict`

```python
def state_dict(self, *, prefix: str = "") -> _STATE_DICT:
    result = {}

    for name, param in self.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(param, torch.Tensor):
            result[_concat_prefix(prefix, name)] = param
        elif isinstance(param, BaseOP):
            result.update(param.state_dict(prefix=_concat_prefix(prefix, name)))

    return result
```

它做的是递归收集权重。

流程：

1. 遍历当前对象的所有实例属性：`self.__dict__.items()`。
2. 如果字段名以 `_` 开头，跳过。
3. 如果字段值是 `torch.Tensor`，加入结果字典。
4. 如果字段值是 `BaseOP`，递归调用它的 `state_dict()`。
5. 其他类型不处理。

举个例子，`Linear` 的定义类似：

```python
class Linear(BaseOP):
    def __init__(...):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None
```

如果 `bias is None`，它不是 Tensor，所以不会进入 `state_dict`。

如果某个模型里有：

```python
self.o_proj = Linear(...)
```

那么 `state_dict(prefix="model.layers.0.self_attn")` 会生成：

```python
{
    "model.layers.0.self_attn.o_proj.weight": tensor(...)
}
```

## `BaseOP.load_state_dict`

```python
for name, param in self.__dict__.items():
    if name.startswith("_"):
        continue
    if isinstance(param, torch.Tensor):
        item = state_dict.pop(_concat_prefix(prefix, name))
        assert isinstance(item, torch.Tensor)
        assert param.shape == item.shape, ...
        setattr(self, name, item)
    elif isinstance(param, BaseOP):
        param.load_state_dict(
            state_dict,
            prefix=_concat_prefix(prefix, name),
            _internal=True,
        )

if not _internal and state_dict:
    raise RuntimeError(...)
```

它和 `state_dict()` 是反向操作。

关键点：

- 通过当前对象结构推导需要哪些 key。
- 从传入的 `state_dict` 里 `pop()` 对应 tensor。
- 检查加载进来的 item 是 `torch.Tensor`。
- 检查 shape 是否一致。
- 用 `setattr(self, name, item)` 替换原来的空 tensor。
- 对子模块递归加载。

这里的 `pop()` 很重要：`load_state_dict()` 会修改传入的字典，把已经消费掉的权重删掉。

最后：

```python
if not _internal and state_dict:
    raise RuntimeError(...)
```

只有最外层调用会检查是否还有未使用的 key。递归调用时 `_internal=True`，避免子模块提前报错，因为字典里还会有其他兄弟模块的权重。

例如 `Qwen3ForCausalLM` 的顶层结构是：

```python
self.model = Qwen3Model(config)
self.lm_head = LMHead(...)
```

加载 `model` 子模块时，`state_dict` 里还会有 `lm_head.*`，所以内部不能马上判定为 unexpected keys。

## `load_state_dict` 的错误行为

这个实现会处理三类错误：

1. 缺少必要 key。

```python
item = state_dict.pop(key)
```

如果 key 不存在，会直接抛 `KeyError`。

2. shape 不一致。

```python
assert param.shape == item.shape
```

会抛 `AssertionError`，错误信息包含 expected/got shape。

注意：`assert` 在 Python 使用 `-O` 优化模式运行时会被禁用。如果这是生产路径，严格来说用显式 `if ...: raise` 会更稳。

3. 多余 key。

```python
raise RuntimeError(f"Unexpected keys in state_dict: ...")
```

只在最外层 `_internal=False` 时检查。

## `StateLessOP`

```python
class StateLessOP(BaseOP):
    def state_dict(self, *, prefix: str = "") -> _STATE_DICT:
        return {}
```

这是“无权重算子”的基类。

典型例子是 `RotaryEmbedding`：

```python
class RotaryEmbedding(StateLessOP):
    ...
    self._cos_cache = ...
    self._sin_cache = ...
```

`RotaryEmbedding` 有缓存 tensor，但这些不是 checkpoint 里的可学习权重，所以字段名以下划线开头，并且继承 `StateLessOP`，`state_dict()` 永远返回 `{}`。

`StateLessOP.load_state_dict()` 也不会加载任何东西，只会在最外层调用时检查是否有多余 key。

## `OPList`

```python
T = TypeVar("T", bound=BaseOP)

class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops
```

`OPList` 是一个专门装 `BaseOP` 的列表容器，类似 PyTorch 的 `nn.ModuleList`。

为什么需要它？

因为 `BaseOP.state_dict()` 只识别两种公开字段：

- `torch.Tensor`
- `BaseOP`

普通 Python `list` 不会被递归处理。所以如果直接写：

```python
self.layers = [Qwen3DecoderLayer(...), ...]
```

这些层的权重会被忽略。

项目里正确写法是：

```python
self.layers = OPList(
    [Qwen3DecoderLayer(config, i) for i in range(config.num_layers)]
)
```

`OPList.state_dict()`：

```python
for i, op in enumerate(self.op_list):
    result.update(op.state_dict(prefix=_concat_prefix(prefix, str(i))))
```

它会用列表下标作为权重路径的一部分。

例如：

```text
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.1.self_attn.qkv_proj.weight
```

这和 HuggingFace checkpoint 里的命名风格是对齐的。

## `OPList.load_state_dict`

```python
for i, op in enumerate(self.op_list):
    op.load_state_dict(
        state_dict,
        prefix=_concat_prefix(prefix, str(i)),
        _internal=True,
    )
```

它按顺序给每个子 OP 加载权重，prefix 中加入下标。

如果 `prefix="model.layers"`，第 0 层会加载：

```text
model.layers.0.*
```

第 1 层会加载：

```text
model.layers.1.*
```

最外层时同样检查是否还有多余 key。

## 和权重加载流程的关系

`base.py` 的 `state_dict()` 不只是保存权重，也被用来“声明模型需要哪些权重”。

在 `python/aios/models/weight.py` 中：

```python
for target_name in model.state_dict():
    ...
    fused_state_dict[target_name] = tensor.to(device=device, dtype=dtype)

model.load_state_dict(fused_state_dict)
```

加载流程是：

1. 先通过 `model.state_dict()` 枚举模型期望的权重 key。
2. 去 safetensors checkpoint 里读取对应 tensor。
3. 对某些 fused 层做拼接，比如 `qkv_proj` 对应 HuggingFace 的 `q_proj/k_proj/v_proj`。
4. 调用 `model.load_state_dict()` 把 tensor 写回模型对象。

所以 `BaseOP.state_dict()` 的 key 规则必须和 checkpoint 命名规则匹配，否则权重加载会失败。

## 和 `torch.nn.Module` 的区别

这个文件实现的是一个极简模块系统，不是 PyTorch 标准 `nn.Module`。

主要区别：

- 不继承 `torch.nn.Module`。
- 不使用 `nn.Parameter`。
- 不支持 PyTorch 自动注册 module/parameter。
- 不支持 buffer 概念。
- 不支持 `modules()`、`parameters()`、`.to()`、`.eval()` 等标准 Module API。
- 权重发现完全依赖 `self.__dict__` 和命名约定。
- 嵌套容器只支持项目自定义的 `OPList`。

优点是简单、透明、适合教学或推理框架控制权重布局。缺点是约束比较强，写子类时必须严格遵守字段命名规则。

## 使用这个基类时的注意事项

写新的 `BaseOP` 子类时应该遵守：

```python
class MyOP(BaseOP):
    def __init__(self):
        self.weight = torch.empty(...)
        self.child = SomeOtherOP(...)
        self._runtime_config = ...
        self._cache = ...

    def forward(self, x):
        ...
```

避免这些写法：

```python
self.layers = [Layer(), Layer()]
```

因为普通 list 不会被递归收集，应使用：

```python
self.layers = OPList([Layer(), Layer()])
```

也要注意：

```python
self.cache = torch.empty(...)
```

如果这是缓存而不是 checkpoint 权重，它会被错误加入 `state_dict`。应命名为：

```python
self._cache = torch.empty(...)
```

## 总结

`python/aios/layers/base.py` 是 AIOS 自己实现的极简模型组件基类：它用“公开 Tensor 是权重、公开 BaseOP 是子模块、下划线字段忽略”的约定，递归生成和加载 checkpoint 权重；`StateLessOP` 表示无权重算子，`OPList` 则补上了列表型子模块的递归管理能力。

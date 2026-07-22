# AIOS `python/aios` 源码阅读顺序指南

这份文档只面向 `/root/aios/python/aios` 下的源码，不讨论课程资源目录中的内容。目标是帮助你从当前代码状态出发，按合理顺序理解 AIOS 这个 LLM 推理引擎的实现。

阅读时建议先建立三条主线：

1. **模型执行线**：token 如何经过 embedding、decoder layer、lm head 变成 logits。
2. **请求调度线**：多个 prompt 如何进入 prefill/decode continuous batching。
3. **显存与 attention 线**：KV cache、page table、FlashInfer metadata、CUDA graph 如何协作。

不要一开始就从 `LLM.generate()` 逐行追到底。当前源码已经包含模型、调度、KV cache、FlashInfer、CUDA graph 等多个层次，直接追完整调用链很容易混淆“数据结构职责”和“运行时执行顺序”。更好的方法是先读基础抽象，再读模型，再读运行时。

## 一、源码目录地图

`python/aios` 当前主要分为这些模块：

```text
python/aios/
├── __init__.py              # 包入口，导出 LLM 和 SamplingParams
├── __main__.py              # 命令行入口
├── core.py                  # 请求、batch、采样参数、全局推理上下文
├── layers/                  # 基础神经网络层和张量辅助函数
├── models/                  # 模型配置、模型工厂、Qwen3 结构、权重加载
├── attention/               # attention backend 抽象和 FlashInfer 实现
├── kvcache/                 # KV cache 存储抽象和 MHA cache 实现
├── scheduler/               # continuous batching 调度、page table、cache page 分配
├── engine/                  # GPU 执行资源、forward、采样、CUDA graph
├── kernel/                  # Triton kernel，目前用于写入 KV cache
└── llm/                     # 用户侧 LLM API，串起 tokenizer、engine、scheduler
```

可以把它们理解为：

```text
llm/        用户 API
engine/     GPU 执行器
scheduler/  请求调度器
models/     模型结构和权重
layers/     模型基础层
attention/  attention 后端
kvcache/    KV cache 存储
kernel/     自定义底层 kernel
core.py     跨模块共享的数据结构和上下文
```

## 二、推荐总阅读顺序

推荐按下面顺序阅读：

1. `python/aios/core.py`
2. `python/aios/layers/base.py`
3. `python/aios/layers/*.py`
4. `python/aios/models/config.py`
5. `python/aios/models/base.py`
6. `python/aios/models/qwen3.py`
7. `python/aios/models/weight.py`
8. `python/aios/models/__init__.py`
9. `python/aios/kvcache/*.py`
10. `python/aios/kernel/store.py`
11. `python/aios/attention/*.py`
12. `python/aios/scheduler/*.py`
13. `python/aios/engine/sample.py`
14. `python/aios/engine/engine.py`
15. `python/aios/engine/graph.py`
16. `python/aios/llm/llm.py`
17. `python/aios/__init__.py` 和 `python/aios/__main__.py`

这个顺序的原因是：先理解共享数据结构和模型结构，再理解运行时如何组织 batch、KV cache 和 attention metadata，最后看用户 API 如何把这些模块串起来。

## 三、第一步：读 `core.py`

先读：

- `python/aios/core.py`

重点看：

- `SamplingParams`
- `Req`
- `Batch`
- `Context`
- `set_global_ctx`
- `clear_global_ctx`
- `get_global_ctx`

这是全项目最重要的共享语义文件。

### `SamplingParams`

`SamplingParams` 描述生成时的采样参数：

```text
temperature
top_k
top_p
ignore_eos
max_tokens
```

其中 `is_greedy` 判断当前是否是贪心采样：

```text
temperature <= 0 或 top_k == 1，并且 top_p == 1.0
```

这个属性会被 `engine/sample.py` 中的 `Sampler` 使用。

### `Req`

`Req` 表示一个正在推理系统中流转的请求。

重点字段：

```text
input_ids        CPU 上的 token 序列
table_idx        该请求占用的 page_table 行号
cached_len       已经写入 KV cache 的 token 数
device_len       当前需要设备侧可见的 token 长度
max_device_len   prompt 长度 + 最大生成长度
generated        已生成 token 列表
```

重点属性：

```text
remain_len = max_device_len - device_len
extend_len = device_len - cached_len
can_decode = remain_len > 0
```

调度器就是靠这些长度判断一个请求该 prefill、decode、完成还是释放资源。

### `Batch`

`Batch` 表示一次模型 forward 的请求集合。

核心字段：

```text
reqs           本次真实参与 forward 的请求
phase          "prefill" 或 "decode"
input_ids      scheduler 填好的设备侧输入 token
positions      scheduler 填好的 position ids
out_loc        KV cache 写入位置
padded_reqs    CUDA graph 场景下补齐后的请求列表
attn_metadata  attention backend 准备好的元数据
```

`Batch.is_prefill` 和 `Batch.is_decode` 会影响：

- `LMHead.forward` 是否只取最后 token。
- `FlashInferBackend` 使用 prefill wrapper 还是 decode wrapper。
- `GraphRunner` 是否可 replay CUDA graph。

### `Context`

`Context` 是模型 forward 时访问运行时状态的桥梁。

它保存：

```text
page_size
page_table
attn_backend
kv_cache
当前 active batch
```

模型层中不会显式传入 KV cache 或 attention backend，而是通过：

```python
get_global_ctx()
```

取得当前上下文。

典型例子：

- `Qwen3Model.forward()` 从 `ctx.batch.input_ids` 取输入。
- `Qwen3Attention.forward()` 调用 `ctx.attn_backend.forward(...)`。
- `LMHead.forward()` 根据 `ctx.batch.is_prefill` 只计算最后 token logits。

理解 `core.py` 后，再读其他模块会容易很多。

## 四、第二步：读 `layers/`

读：

- `python/aios/layers/base.py`
- `python/aios/layers/linear.py`
- `python/aios/layers/embedding.py`
- `python/aios/layers/norm.py`
- `python/aios/layers/rotary.py`
- `python/aios/layers/attention.py`
- `python/aios/layers/activation.py`
- `python/aios/layers/__init__.py`

这一层定义模型内部用到的基础组件。

### `layers/base.py`

重点看：

- `_concat_prefix`
- `BaseOP.state_dict`
- `BaseOP.load_state_dict`
- `StateLessOP`
- `OPList`

必须掌握这条规则：

```text
公开 torch.Tensor 字段 -> 权重
公开 BaseOP 字段       -> 子模块，递归遍历
以下划线开头字段       -> 运行时状态、cache、配置，不进入 state_dict
```

这套机制替代了 PyTorch `nn.Module` 的参数注册系统。

例如：

```text
model.layers.0.self_attn.qkv_proj.weight
```

这个 key 是由对象属性名递归拼出来的，不是手写表维护的。

### `layers/linear.py`

重点看：

- `Linear`
- `LinearQKVMerged`
- `LinearColParallelMerged`

`Linear` 是基础线性层，使用：

```python
F.linear(x, self.weight, self.bias)
```

`LinearQKVMerged` 把 attention 的 Q/K/V 投影合并成一个权重：

```text
[Q | K | V]
```

`LinearColParallelMerged` 把 MLP 的 gate/up 投影合并成：

```text
[gate | up]
```

这两个 merged 类和 `models/weight.py` 中的 checkpoint 权重拼接逻辑直接对应。

### `layers/embedding.py`

重点看：

- `Embedding`
- `LMHead`

`Embedding` 负责 token id 查表。

`LMHead` 负责 hidden state 到 vocab logits 的投影，同时支持：

- `tie_word_embeddings=True` 时复用输入 embedding 权重。
- prefill 阶段只取每条请求最后 token 的 hidden state 计算 logits。

这一点对性能很关键，因为 prompt 中间 token 的完整词表 logits 通常不需要。

### `layers/norm.py`

重点看：

- `RMSNorm`
- `RMSNorm.forward_inplace`
- `RMSNormFused`

这里使用 FlashInfer 的：

```text
rmsnorm
fused_add_rmsnorm
```

`RMSNorm` 用于 Q/K norm。

`RMSNormFused` 用于 Transformer block 中 residual add 和 RMSNorm 的融合路径。

### `layers/rotary.py` 和 `layers/attention.py`

`rotary.py` 负责预计算 RoPE 的 cos/sin cache：

```text
_cos_cache
_sin_cache
```

它继承 `StateLessOP`，说明这些 cache 不是 checkpoint 权重。

`attention.py` 提供 RoPE 和 GQA/MQA 相关的无状态函数：

- `rotate_half`
- `apply_rotary_pos_emb`
- `repeat_kv`

### `layers/activation.py`

重点看：

- `silu_and_mul`

它封装 FlashInfer 的 fused SwiGLU 激活：

```text
silu(gate) * up
```

对应 Qwen3 MLP 中的 `gate_up_proj` 输出布局 `[gate | up]`。

### `layers/__init__.py`

这个文件统一导出 layer API，让模型代码可以写：

```python
from aios.layers import Linear, RMSNorm, OPList
```

它是 `layers` 包对上层暴露的接口边界。

## 五、第三步：读 `models/`

读：

- `python/aios/models/config.py`
- `python/aios/models/base.py`
- `python/aios/models/qwen3.py`
- `python/aios/models/weight.py`
- `python/aios/models/__init__.py`

这一层回答两个问题：

1. 模型结构是什么。
2. checkpoint 权重如何加载进自定义 `BaseOP` 模型。

### `models/config.py`

重点看 `ModelConfig`。

它把 HuggingFace config 或 `config.json` 中的字段转成 AIOS 模型需要的最小配置集合：

```text
num_layers
num_qo_heads
num_kv_heads
head_dim
hidden_size
vocab_size
intermediate_size
hidden_act
rms_norm_eps
rope_theta
max_position_embeddings
tie_word_embeddings
```

读这个文件时要把字段和 `Qwen3Attention/Qwen3MLP/Qwen3Model` 的初始化参数对应起来。

### `models/base.py`

`BaseLLMModel` 同时继承：

```text
ABC
BaseOP
```

它只规定模型必须实现：

```python
forward() -> torch.Tensor
```

### `models/qwen3.py`

建议按类顺序读：

1. `Qwen3Attention`
2. `Qwen3MLP`
3. `Qwen3DecoderLayer`
4. `Qwen3Model`
5. `Qwen3ForCausalLM`

#### `Qwen3Attention`

核心流程：

```text
hidden_states
-> qkv_proj
-> split Q/K/V
-> reshape 成 head 视图
-> q_norm/k_norm
-> apply_rotary_pos_emb
-> ctx.attn_backend.forward
-> o_proj
```

注意：attention 的实际计算不在 `qwen3.py` 中手写，而是交给 `ctx.attn_backend`。这就是模型结构和 attention 后端解耦的关键。

#### `Qwen3MLP`

核心流程：

```text
x
-> gate_up_proj
-> silu_and_mul
-> down_proj
```

其中 `gate_up_proj` 是 merged projection，输出 `[gate | up]`。

#### `Qwen3DecoderLayer`

核心流程：

```text
input_layernorm
-> self_attn
-> post_attention_layernorm
-> mlp
```

`RMSNormFused` 同时处理 residual 路径。

#### `Qwen3Model`

核心流程：

```text
ctx.batch.input_ids
-> embed_tokens
-> rotary_emb(batch.positions)
-> 遍历 layers
-> final norm
```

注意 `_rotary_emb` 以下划线开头，不进入 `state_dict`。

#### `Qwen3ForCausalLM`

顶层语言模型：

```text
Qwen3Model
-> LMHead
```

它返回 logits，后续由 `Engine.forward_batch()` 调用 sampler 生成 next token。

### `models/weight.py`

重点看：

- `packed_modules_mapping`
- `_checkpoint_index`
- `_read_tensor`
- `_packed_source_names`
- `load_weights`

权重加载流程：

```text
扫描 safetensors 文件
-> 建立 checkpoint key 到文件路径的索引
-> 遍历 model.state_dict() 需要的目标 key
-> 直接读取或拼接 HF 源权重
-> 移动到目标 device/dtype
-> model.load_state_dict(fused_state_dict)
```

`packed_modules_mapping` 处理两个重要映射：

```text
qkv_proj      <- q_proj + k_proj + v_proj
gate_up_proj <- gate_proj + up_proj
```

这就是为什么模型里可以使用 fused layer，而 checkpoint 仍然兼容 HuggingFace 的拆分权重。

### `models/__init__.py`

重点看：

- `create_model`

当前通过 `model_path.lower()` 判断是否包含 `"qwen3"`，然后创建 `Qwen3ForCausalLM`。

这是模型工厂入口。将来支持更多模型时，通常会从这里扩展。

## 六、第四步：读 `kvcache/` 和 `kernel/`

读：

- `python/aios/kvcache/base.py`
- `python/aios/kvcache/mha_pool.py`
- `python/aios/kvcache/naive_manager.py`
- `python/aios/kvcache/__init__.py`
- `python/aios/kernel/store.py`

这一层负责 KV cache 的物理存储和写入。

### `kvcache/base.py`

重点看三个抽象：

- `BaseKVCache`
- `BaseCacheHandle`
- `BaseCacheManager`

`BaseKVCache` 定义真实 KV cache 存储必须提供：

```text
k_cache(layer_id)
v_cache(layer_id)
store_kv(k, v, out_loc, layer_id)
device
dtype
num_layers
```

`BaseCacheManager` 是 prefix cache 或 cache 管理策略的接口。当前主运行路径里实际使用的是 `scheduler/cache.py` 的 page 分配器，而 `NaiveCacheManager` 更像一个简单实现和接口占位。

### `kvcache/mha_pool.py`

`MHAKVCache` 是真实 KV cache 存储。

核心 buffer 形状：

```text
(2, num_layers, num_pages, page_size, num_kv_heads, head_dim)
```

第一维 `2` 分别表示：

```text
0 -> K cache
1 -> V cache
```

`store_kv` 会调用：

```python
from aios.kernel import store_cache
```

把当前 batch 算出来的 K/V 写入物理 cache 位置。

### `kernel/store.py`

这个文件定义 Triton kernel：

- `_store_cache_kernel`
- `store_cache`

`store_cache` 做输入检查，然后 launch Triton kernel，把扁平 token batch 的 K/V scatter 到 cache 中。

输入关系：

```text
k/v       当前 forward 新算出来的 K/V
indices   每个 token 应写入的物理 cache slot
k_cache   某一层的 K cache
v_cache   某一层的 V cache
```

理解这个文件后，就能明白 attention backend 不是只读 cache，它也会在每层 attention 前先写入当前 token 的 K/V。

## 七、第五步：读 `attention/`

读：

- `python/aios/attention/base.py`
- `python/aios/attention/utils.py`
- `python/aios/attention/fi.py`
- `python/aios/attention/__init__.py`

这一层负责把模型产生的 Q/K/V 接到具体 attention backend。

### `attention/base.py`

重点看：

- `BaseAttnMetadata`
- `BaseAttnBackend`
- `HybridBackend`

`BaseAttnBackend` 定义 attention 后端必须实现：

```text
forward
prepare_metadata
init_capture_graph
prepare_for_capture
prepare_for_replay
```

这说明 attention backend 不只是算 attention，还负责为 FlashInfer 或 CUDA graph 准备运行元数据。

`HybridBackend` 可以根据 batch phase 在 prefill 和 decode 使用不同 backend。

### `attention/utils.py`

`BaseCaptureData` 是 CUDA graph capture/replay 使用的静态 buffer 数据结构。

它保存：

```text
seq_lens
positions
cu_seqlens_k
cu_seqlens_q
page_table
```

这些字段后续会被 FlashInfer graph wrapper 使用。

### `attention/fi.py`

这是 FlashInfer attention backend。

重点看：

- `FICaptureData`
- `FIMetadata`
- `FlashInferBackend.__init__`
- `FlashInferBackend.prepare_metadata`
- `FlashInferBackend.forward`
- `FlashInferBackend.init_capture_graph`
- `FlashInferBackend.prepare_for_capture`
- `FlashInferBackend.prepare_for_replay`

#### `FIMetadata`

它保存一次 batch attention 所需的 FlashInfer plan 参数：

```text
cu_seqlens_q_cpu
cu_seqlens_k_cpu
cu_seqlens_q_gpu
indices
last_page_len_cpu
seq_lens_cpu
num_qo_heads
num_kv_heads
head_dim
wrapper
```

`get_last_indices` 会被 `LMHead.forward()` 使用，用于 prefill 阶段提取每条序列最后 token 的 hidden state。

#### `prepare_metadata`

根据 `batch.padded_reqs` 构建：

```text
query lengths
key lengths
cached lengths
page table indices
FlashInfer wrapper
```

prefill batch 使用 prefill wrapper，decode batch 使用 decode wrapper。

#### `forward`

核心流程：

```text
初始化 FlashInfer plan
-> store_kv 写入当前层 K/V
-> 取出该层完整 KV cache
-> wrapper.run(q, paged_kv_cache)
```

这个函数是模型 attention 和 KV cache/FlashInfer 真正汇合的地方。

## 八、第六步：读 `scheduler/`

读：

- `python/aios/scheduler/common.py`
- `python/aios/scheduler/table.py`
- `python/aios/scheduler/cache.py`
- `python/aios/scheduler/prefill.py`
- `python/aios/scheduler/decode.py`
- `python/aios/scheduler/scheduler.py`
- `python/aios/scheduler/__init__.py`

这一层负责 continuous batching。

### `scheduler/common.py`

`PendingReq` 表示还没进入 prefill 的请求。

它保存：

```text
uid
input_ids
sampling_params
```

并提供：

```text
input_len
output_len
```

### `scheduler/table.py`

`TableManager` 管理请求槽位和 token pool。

核心字段：

```text
page_table   GPU int32 表，记录请求每个 logical position 对应的物理 cache slot
token_pool   GPU int32 表，记录请求 token id
_free_slots  可用请求槽位
```

`table_idx` 是请求在 `page_table/token_pool` 中占用的行号。

### `scheduler/cache.py`

`CacheManager` 管理物理 KV page 的分配和释放。

核心流程：

```text
allocate_paged(reqs)
-> 根据 req.cached_len 和 req.device_len 计算需要新增的 page
-> 从 free_slots 分配物理位置
-> 写入 page_table
```

`free_req` 根据请求已经缓存的 token 释放对应物理 page。

当前 `Engine` 里 `Context(page_size=1)`，所以一个 page 对应一个 token slot。

### `scheduler/prefill.py`

`PrefillManager` 管理等待进入系统的新请求。

核心逻辑：

- `pending_list` 保存待处理请求。
- `_can_admit` 判断 table slot 和 KV cache 空间是否足够。
- `schedule_next_batch` 按 prefill token budget 选一批 pending 请求。
- 选中的 prompt token 会先写入 `TableManager.token_pool`。
- 返回 `Batch(reqs=reqs, phase="prefill")`。

### `scheduler/decode.py`

`DecodeManager` 管理已经完成 prefill、正在逐 token decode 的请求集合。

核心逻辑：

- `running_reqs` 保存活跃请求。
- `schedule_next_batch` 把所有可 decode 请求组成一个 decode batch。
- `filter_reqs` 把新请求加入 running set，并移除不能继续 decode 的请求。
- `inflight_tokens` 用于估算已运行请求未来还需要的 KV cache 空间。

### `scheduler/scheduler.py`

这是调度器总入口。

重点看：

- `ForwardInput`
- `Scheduler.add_request`
- `Scheduler.schedule_next_batch`
- `Scheduler._prepare_batch`
- `Scheduler.process_batch_output`
- `Scheduler.collect_results`
- `_make_positions`
- `_make_input_tuple`
- `_make_write_tuple`

调度策略是 prefill-first：

```text
优先从 PrefillManager 调度 prefill batch
如果没有 prefill batch，再从 DecodeManager 调度 decode batch
```

`_prepare_batch` 是 scheduler 和模型 forward 之间的关键桥梁：

```text
可能做 CUDA graph padding
-> 分配 KV page
-> 构造 positions
-> 构造 input_mapping
-> 构造 write_mapping
-> batch.out_loc = page_table[input_mapping]
-> attn_backend.prepare_metadata(batch)
```

返回的 `ForwardInput` 会被 `LLM.generate()` 使用：

```text
batch.input_ids = table_manager.token_pool[input_tuple]
next_tokens = engine.forward_batch(batch)
scheduler.process_batch_output(...)
```

## 九、第七步：读 `engine/`

读：

- `python/aios/engine/sample.py`
- `python/aios/engine/engine.py`
- `python/aios/engine/graph.py`
- `python/aios/engine/__init__.py`

这一层拥有 GPU 执行资源。

### `engine/sample.py`

`Sampler` 根据 logits 生成 next token。

路径：

```text
greedy -> argmax
非 greedy -> temperature -> top_k -> softmax -> multinomial
```

当前 `SamplingParams.top_p` 只参与 `is_greedy` 判断，没有实现 nucleus sampling 过滤。阅读时要注意这一点。

### `engine/engine.py`

`Engine` 是运行时核心。

初始化阶段做这些事：

```text
设置 CUDA device 和 stream
meta device 上创建模型结构
加载 checkpoint 权重
移动 RoPE cache 到 device
根据剩余显存计算 KV cache pages
创建 Context
创建 MHAKVCache
创建 page_table
设置 global ctx
创建 FlashInferBackend
创建 dummy req/page
按需创建 GraphRunner
```

`forward_batch` 是一次 batch 推理：

```text
with ctx.forward_batch(batch)
-> 如果可用 CUDA graph，graph_runner.replay(batch)
-> 否则 model.forward()
-> req.complete_one()
-> 取真实 batch logits
-> Sampler 采样 next token
-> 返回 int32 next_tokens
```

注意：模型 forward 本身不接收 batch 参数，它通过 global context 获取当前 batch。

### `engine/graph.py`

`GraphRunner` 管理 decode 阶段 CUDA graph capture/replay。

重点看：

- `GraphCaptureBuffer`
- `determine_cuda_graph_bs`
- `GraphRunner._capture_graphs`
- `GraphRunner.can_use_cuda_graph`
- `GraphRunner.pad_batch`
- `GraphRunner.replay`

核心思想：

```text
decode batch shape 相对固定
-> 预先为多个 batch size bucket capture CUDA graph
-> 运行时把 batch pad 到 bucket size
-> 拷贝 input_ids/out_loc/positions 到静态 buffer
-> replay 对应 graph
```

dummy request 用于补齐 batch size，补齐部分 logits 后续会被切掉。

## 十、第八步：读 `llm/`

读：

- `python/aios/llm/llm.py`
- `python/aios/llm/__init__.py`

这是用户侧 API。

### `LLM.__init__`

初始化流程：

```text
规范化 CUDA device
解析本地路径或下载 HuggingFace snapshot
读取 HF config
转成 ModelConfig
加载 tokenizer
解析 CUDA graph 参数
创建 Engine
创建 CacheManager
```

这里的 `CacheManager` 是 scheduler 使用的 page 分配器；真实 KV cache 存储在 `Engine.kv_cache`。

### `LLM.generate`

这是用户调用生成的入口。

核心流程：

```text
规范化 sampling_params
-> prompt 文本经过 tokenizer/chat template 得到 input_ids
-> 检查 max_total_len 是否超过 engine.max_seq_len
-> reset_page_table
-> 创建 TableManager
-> 创建 Scheduler
-> add_request
-> while scheduler.has_work:
       forward_input = scheduler.schedule_next_batch()
       batch.input_ids = table_manager.token_pool[forward_input.input_tuple]
       next_tokens = engine.forward_batch(batch)
       scheduler.process_batch_output(forward_input, next_tokens)
-> collect_results
```

读完这个函数后，你应该能把前面的模块串成完整调用链。

## 十一、第九步：读包入口和 CLI

读：

- `python/aios/__init__.py`
- `python/aios/__main__.py`

`__init__.py` 通常用于导出用户 API，例如：

```python
from aios import LLM, SamplingParams
```

`__main__.py` 是命令行入口，用于从终端启动一次生成流程。

这两个文件适合最后读，因为它们只是把前面模块组合成对外接口。

## 十二、主调用链速查

从用户调用到模型 forward 的主路径：

```text
LLM.generate
-> Scheduler.add_request
-> Scheduler.schedule_next_batch
-> Scheduler._prepare_batch
-> FlashInferBackend.prepare_metadata
-> Engine.forward_batch
-> Context.forward_batch(batch)
-> Qwen3ForCausalLM.forward
-> Qwen3Model.forward
-> Qwen3DecoderLayer.forward
-> Qwen3Attention.forward
-> FlashInferBackend.forward
-> MHAKVCache.store_kv
-> store_cache Triton kernel
-> FlashInfer wrapper.run
-> LMHead.forward
-> Sampler.sample
-> Scheduler.process_batch_output
-> Scheduler.collect_results
```

从 checkpoint 到模型权重的路径：

```text
LLM.__init__
-> Engine.__init__
-> create_model
-> load_weights
-> model.state_dict()
-> safetensors 读取或 fused 拼接
-> model.load_state_dict()
```

从请求到 KV cache 位置的路径：

```text
PendingReq
-> PrefillManager.schedule_next_batch
-> TableManager.allocate
-> Scheduler._prepare_batch
-> CacheManager.allocate_paged
-> page_table 写入物理 slot
-> batch.out_loc = page_table[input_mapping]
-> MHAKVCache.store_kv(k, v, out_loc, layer_id)
```

## 十三、按目标反查阅读路径

### 想理解模型结构

读：

1. `layers/base.py`
2. `layers/linear.py`
3. `layers/embedding.py`
4. `layers/norm.py`
5. `layers/rotary.py`
6. `layers/attention.py`
7. `models/config.py`
8. `models/qwen3.py`

重点问题：

- 权重 key 如何生成？
- Q/K/V 如何从 fused projection 中拆出来？
- RoPE 在哪里生成，在哪里应用？
- attention 为什么交给 backend？
- MLP 的 `[gate | up]` 如何计算？

### 想理解权重加载

读：

1. `layers/base.py`
2. `models/qwen3.py`
3. `models/weight.py`
4. `models/__init__.py`
5. `engine/engine.py`

重点问题：

- `model.state_dict()` 枚举了哪些目标 key？
- `qkv_proj` 和 `gate_up_proj` 如何从 HF 拆分权重拼接？
- 为什么模型先在 `meta` device 上创建？
- RoPE cache 为什么需要单独 `set_device`？

### 想理解 continuous batching

读：

1. `core.py`
2. `scheduler/common.py`
3. `scheduler/table.py`
4. `scheduler/cache.py`
5. `scheduler/prefill.py`
6. `scheduler/decode.py`
7. `scheduler/scheduler.py`
8. `llm/llm.py`

重点问题：

- pending/running/finished 三类请求如何流转？
- prefill 和 decode batch 如何切换？
- `cached_len/device_len/extend_len` 分别表示什么？
- token_pool 和 page_table 分别解决什么问题？

### 想理解 KV cache 和 attention backend

读：

1. `kvcache/base.py`
2. `kvcache/mha_pool.py`
3. `kernel/store.py`
4. `attention/base.py`
5. `attention/fi.py`
6. `models/qwen3.py` 中的 `Qwen3Attention`

重点问题：

- K/V cache 的物理 shape 是什么？
- 当前 token 的 K/V 什么时候写入 cache？
- `batch.out_loc` 从哪里来？
- FlashInfer metadata 如何描述变长 prefill 和 decode？

### 想理解 CUDA graph

读：

1. `engine/graph.py`
2. `attention/fi.py` 中 graph capture/replay 相关方法
3. `scheduler/scheduler.py` 中 `_prepare_batch`
4. `engine/engine.py` 中 `forward_batch`

重点问题：

- 哪些 batch 可以用 CUDA graph？
- 为什么要 pad batch？
- 静态 buffer 保存了哪些输入？
- attention backend 在 capture 和 replay 前分别准备什么？

### 想理解用户 API

读：

1. `llm/llm.py`
2. `engine/engine.py`
3. `scheduler/scheduler.py`
4. `core.py`

重点问题：

- prompt 如何转成 input_ids？
- sampling params 如何绑定到每个请求？
- generate 循环何时停止？
- 最终结果如何按 uid 排序返回？

## 十四、几个关键心智模型

### 1. 模型 forward 不显式接收 batch

`Qwen3ForCausalLM.forward()` 没有参数。

它依赖：

```python
get_global_ctx()
```

取得当前 batch、KV cache、attention backend 等运行时状态。

这让模型结构代码保持简单，但也意味着模型 forward 必须在：

```python
with ctx.forward_batch(batch):
```

上下文中调用。

### 2. `page_table` 是逻辑位置到物理 KV slot 的映射

请求有自己的 logical position：

```text
0, 1, 2, ...
```

KV cache 里有全局物理 slot。

`page_table[table_idx, position]` 记录某个请求某个位置对应哪个物理 slot。

attention backend 和 `store_cache` 都依赖这个映射。

### 3. `token_pool` 保存设备侧 token id

`Req.input_ids` 是 CPU tensor。

模型 forward 需要 CUDA 上的 `input_ids`。

`TableManager.token_pool` 是 GPU 上的 token 表，scheduler 通过 input mapping 取出本次 batch 需要的 token：

```python
batch.input_ids = table_manager.token_pool[forward_input.input_tuple]
```

### 4. prefill 和 decode 的区别来自 `Req` 长度

prefill：

```text
cached_len = 0
device_len = prompt_len
extend_len = prompt_len
```

decode：

```text
cached_len = 已缓存长度
device_len = cached_len + 1
extend_len = 1
```

FlashInfer metadata 会根据这些长度构造不同的 attention plan。

### 5. fused layer 的 checkpoint 适配放在加载器

模型里使用：

```text
qkv_proj
gate_up_proj
```

checkpoint 里通常是：

```text
q_proj/k_proj/v_proj
gate_proj/up_proj
```

这个差异由 `models/weight.py` 解决，而不是由 layer 或模型 forward 解决。

## 十五、建议的实际阅读方式

第一遍只读结构：

```text
core.py
layers/base.py
models/qwen3.py
llm/llm.py
```

目标是知道有哪些对象，以及它们如何互相调用。

第二遍读运行时：

```text
scheduler/scheduler.py
scheduler/prefill.py
scheduler/decode.py
scheduler/cache.py
engine/engine.py
attention/fi.py
kvcache/mha_pool.py
kernel/store.py
```

目标是理解一次 batch forward 前后，token、position、page table、KV cache 如何变化。

第三遍读优化细节：

```text
engine/graph.py
attention/fi.py
layers/norm.py
layers/activation.py
layers/linear.py
```

目标是理解 FlashInfer、fused ops、CUDA graph 和 fused weights 为什么能提高推理效率。

## 十六、结论

当前 `python/aios` 源码可以按四层理解：

```text
用户入口层：llm/
执行调度层：engine/ + scheduler/
模型计算层：models/ + layers/
显存后端层：attention/ + kvcache/ + kernel/
```

推荐阅读路线是先掌握 `core.py` 的数据结构，再理解 `layers` 和 `models` 的模型 forward，随后进入 `scheduler`、`kvcache`、`attention` 和 `engine` 的运行时协作，最后回到 `LLM.generate()` 串起完整调用链。

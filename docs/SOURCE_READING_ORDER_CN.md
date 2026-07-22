# AIOS 源码阅读顺序指南

这份文档用于从零阅读当前仓库源码。仓库里的 `python/aios/` 已经包含后续课程优化，不只是 Lesson 3 的最小包化版本；因此阅读时建议把两条线分开：

1. 先按课程文档建立概念和演进顺序。
2. 再按源码模块理解当前实现。

不要一开始直接从 `LLM.generate()` 往下追完整调用链。当前主线已经包含 KV cache、paged cache、continuous batching、FlashInfer、fused layers、CUDA graphs 等内容，直接追会很容易把 Lesson 3 的重构目标和后续系统优化混在一起。

## 一、必读 Markdown 顺序

### 0. 项目总览

先读：

- `README.md`

重点看：

- `The Key Mental Model: LLM Inference Engine as an "Operating System"`
- `Performance Progression`
- `Course Roadmap`
- `Engine Architecture (Final State)`

这一步只需要建立整体地图：AIOS 不是单纯写一个模型，而是在逐步构建一个 LLM 推理引擎。后面的 scheduler、KV cache、attention backend、CUDA graph 都是围绕“让 GPU 更高效地跑推理”展开。

### 1. Lesson 0：推理引擎的心智模型

读：

- `resources/lesson-0-introduction/README_CN.md`

重点看：

- `Part 3: 计算和内存：根本性差异`
- `Part 4: 硬件平台：CPU vs GPU`
- `Part 5: 推理引擎 = LLM 的"操作系统"`
- `学习路径`

这一步回答的是“为什么需要推理引擎”。后面所有优化都不是为了让代码更复杂，而是为了处理 GPU 利用率、显存管理、请求调度和吞吐问题。

### 2. Lesson 1：LLM 基础算子

读：

- `resources/lesson-1-llm-basics/README_CN.md`

重点看：

- `Tokenization：text -> tokens -> IDs`
- `Embedding：token ID -> 向量`
- `LayerNorm 与 RMSNorm：层归一化`
- `Q/K/V 生成（Linear 投影）`
- `RoPE：旋转位置编码`
- `Softmax 函数：概率归一化`
- `Attention：注意力机制`
- `MLP（前馈网络）`
- `完整的 Transformer 层`

这一步要把模型 forward 的数学结构看懂。后面阅读 `python/aios/layers/` 和 `python/aios/models/qwen3.py` 时，这些概念会一一对应到代码。

### 3. Lesson 2：单文件 Qwen3 推理

读：

- `resources/lesson-2-run-qwen3/README.md`
- `resources/lesson-2-run-qwen3/run_qwen3.py`

重点看 `run_qwen3.py` 里的这些类和函数：

- `RMSNorm`
- `RotaryEmbedding`
- `repeat_kv`
- `apply_rotary_pos_emb`
- `Qwen3Attention`
- `Qwen3MLP`
- `Qwen3DecoderLayer`
- `Qwen3Model`
- `Qwen3ForCausalLM`
- `load_weights_from_hf`
- `generate`

这一步的目标是理解最朴素的端到端流程：

```text
prompt text
-> tokenizer
-> input_ids
-> embedding
-> N 层 decoder
-> final norm
-> lm_head
-> logits
-> sample/argmax
-> 追加 next token
-> 重复
```

Lesson 2 是后续所有重构和优化的基准版本。看不懂 Lesson 2，直接看 `python/aios/` 会很吃力。

### 4. Lesson 3：从单文件到 `aios` 包

重点读：

- `resources/lesson-3-refactor-to-package/README.md`

这是阅读当前源码的核心文档。重点看：

- `改造前后对比`
- `总览：目录结构`
- `文件依赖关系`
- `第 1 步：创建 BaseOP 基类替换 nn.Module`
- `第 2 步：替换 nn.Linear`
- `第 4 步：替换 RotaryEmbedding`
- `第 6 步：替换 Embedding 和 LMHead`
- `第 10 步：迁移模型实现`
- `第 11 步：safetensors 直接加载`
- `safetensors Key 与 BaseOP state_dict 对应关系`
- `踩坑记录`

Lesson 3 的核心不是新增推理能力，而是重构架构：

```text
run_qwen3.py 单文件
-> python/aios/layers/
-> python/aios/models/
-> python/aios/engine/
-> python/aios/llm/
-> 可复用推理包
```

最重要的概念是 `BaseOP`：它用非常简单的规则替代 `nn.Module` 的参数管理。

## 二、Lesson 3 源码阅读顺序

如果当前目标是理解“从单文件到推理框架”，请按下面顺序读源码。

### 1. 先读基础权重系统

读：

- `python/aios/layers/base.py`

重点看：

- `_concat_prefix`
- `BaseOP.state_dict`
- `BaseOP.load_state_dict`
- `StateLessOP`
- `OPList`

必须理解这条规则：

```text
不以 _ 开头的 torch.Tensor 属性 -> 模型权重
不以 _ 开头的 BaseOP 属性       -> 子模块，递归进入
以 _ 开头的属性                 -> 运行时状态、cache、配置，不进入 state_dict
```

例如：

```text
Qwen3ForCausalLM.model.layers.0.self_attn.q_proj.weight
```

这个 key 不是手写出来的，而是由对象属性名递归拼出来的。它能和 safetensors 里的权重名对齐，是 Lesson 3 能成立的根本原因。

### 2. 再读基础层

读：

- `python/aios/layers/linear.py`
- `python/aios/layers/embedding.py`
- `python/aios/layers/norm.py`
- `python/aios/layers/rotary.py`
- `python/aios/layers/attention.py`
- `python/aios/layers/activation.py`
- `python/aios/layers/__init__.py`

重点对应关系：

```text
nn.Linear      -> Linear
nn.Embedding   -> Embedding
lm_head Linear -> LMHead
RMSNorm        -> RMSNorm / RMSNormFused
RoPE buffer    -> RotaryEmbedding 的 _cos_cache / _sin_cache
纯函数          -> attention.py / activation.py
```

注意：当前 `linear.py` 里已经有 `LinearQKVMerged` 和 `LinearColParallelMerged`，这是 Lesson 9 fused layers 的内容。理解 Lesson 3 时，先只看 `Linear` 本身即可。

当前 `norm.py` 使用了 FlashInfer 的 `rmsnorm` 和 `fused_add_rmsnorm`，这也是后续优化。Lesson 3 的基础版可以理解成：

```python
x_float = x.float()
rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + eps)
return (x_float * rms).to(x.dtype) * weight
```

### 3. 再读模型配置和模型工厂

读：

- `python/aios/models/config.py`
- `python/aios/models/base.py`
- `python/aios/models/__init__.py`

重点看：

- `ModelConfig`
- `ModelConfig.from_json`
- `ModelConfig.from_hf`
- `BaseLLMModel`
- `create_model`

Lesson 3 文档强调的是 `from_json`：直接读取模型目录下的 `config.json`，减少对 `transformers.AutoConfig` 的依赖。

当前 `llm/llm.py` 仍使用了 `AutoConfig.from_pretrained`，再转成 `ModelConfig.from_hf`。这是当前主线为了兼容 HuggingFace Hub 的实际实现。阅读时要知道它和 Lesson 3 的“直读 JSON”目标略有差异。

### 4. 再读 Qwen3 模型结构

读：

- `python/aios/models/qwen3.py`

建议按类顺序读：

1. `Qwen3Attention`
2. `Qwen3MLP`
3. `Qwen3DecoderLayer`
4. `Qwen3Model`
5. `Qwen3ForCausalLM`

与 Lesson 2 对照：

```text
run_qwen3.py: Qwen3Attention(nn.Module)
aios:         Qwen3Attention(BaseOP)

run_qwen3.py: nn.Linear
aios:         Linear / merged Linear

run_qwen3.py: nn.ModuleList
aios:         OPList

run_qwen3.py: layer(...)
aios:         layer.forward(...)
```

注意：当前 `Qwen3Attention.forward` 里通过 `get_global_ctx()` 调用 `ctx.attn_backend.forward(...)`，这是 Lesson 8 之后引入 attention backend 的写法。Lesson 3 的基础版 attention 仍然是直接 `Q @ K.T -> softmax -> @ V`。

### 5. 再读权重加载

读：

- `python/aios/models/weight.py`

重点看：

- `_checkpoint_index`
- `_read_tensor`
- `load_weights`
- `packed_modules_mapping`

Lesson 3 的目标是：

```text
不再 AutoModelForCausalLM.from_pretrained(...)
不再先加载一份 HF 模型再复制 state_dict
而是直接 safetensors -> model.load_state_dict(...)
```

当前实现还处理了 fused 权重：

```text
q_proj + k_proj + v_proj -> qkv_proj
gate_proj + up_proj      -> gate_up_proj
```

这是 Lesson 9 的内容。只看 Lesson 3 时，你可以先把它理解成“权重加载边界处做了一次名称和布局转换”。

### 6. 最后读最小 API 层

读：

- `python/aios/core.py`
- `python/aios/engine/sample.py`
- `python/aios/__init__.py`
- `python/aios/__main__.py`

重点看：

- `SamplingParams`
- `Sampler.sample`
- 包导出入口
- CLI 参数如何变成 `LLM(...).generate(...)`

这部分对应 Lesson 3 中把 `generate()` 函数和 `main()` 函数封装成用户接口。

## 三、当前完整源码阅读顺序

如果你已经读完 Lesson 3，想继续读当前主线实现，建议按下面顺序。

### 1. 运行入口

读：

- `python/aios/__main__.py`
- `python/aios/llm/llm.py`

重点看：

- CLI 参数如何构造 `LLM`
- `LLM.__init__` 如何初始化 tokenizer、engine、cache manager
- `LLM.generate` 如何创建 scheduler
- prompt 如何被 tokenizer 变成 `input_ids`

不要一开始深追所有 scheduler 细节，只需要先看出顶层流程。

### 2. 核心数据结构

读：

- `python/aios/core.py`

重点看：

- `SamplingParams`
- `Req`
- `Batch`
- `Context`
- `set_global_ctx`
- `get_global_ctx`

当前源码里，模型 forward 不再显式传一堆参数，而是通过全局 `Context` 获取当前 batch、attention backend、page table、KV cache 等运行时状态。

### 3. Engine 层

读：

- `python/aios/engine/engine.py`
- `python/aios/engine/sample.py`
- `python/aios/engine/graph.py`

重点看：

- model 如何创建
- 权重如何加载
- attention backend 如何初始化
- KV cache 如何初始化
- `forward_batch` 如何包住模型 forward
- CUDA graph decode replay 如何接入

这部分是“真正执行一次 forward”的中控层。

### 4. KV cache 层

先读文档：

- `resources/lesson-4-kv-cache/README_CN.md`
- `resources/lesson-5-paged-kv-cache/README_CN.md`
- `resources/lesson-5-paged-kv-cache/FEATURE_DESIGN.md`

再读源码：

- `python/aios/kvcache/base.py`
- `python/aios/kvcache/mha_pool.py`
- `python/aios/kvcache/naive_manager.py`
- `python/aios/scheduler/cache.py`
- `python/aios/scheduler/table.py`

重点理解：

- Prefill 和 Decode 为什么要分开
- 每层 K/V 为什么可以缓存
- page table 如何把逻辑 token 位置映射到物理 cache page
- `TableManager` 管 token 和 page 表
- `CacheManager` 管 page 分配和释放

### 5. Attention backend 层

先读文档：

- `resources/lesson-8-flat-varlen-prefill/README_CN.md`

再读源码：

- `python/aios/attention/base.py`
- `python/aios/attention/fi.py`
- `python/aios/attention/utils.py`

重点理解：

- flat token 表示法
- `qo_indptr`、`kv_indptr`、`kv_indices` 这类 metadata
- prefill 和 decode 分别走什么 attention wrapper
- 为什么 attention backend 要从模型层里抽出来

### 6. Scheduler 层

先读文档：

- `resources/lesson-6-static-batching/TECHNICAL_CN.md`
- `resources/lesson-7-continuous-batching/README_CN.md`
- `resources/lesson-7-continuous-batching/TECHNICAL_CN.md`
- `resources/lesson-8-flat-varlen-prefill/README_CN.md`

再读源码：

- `python/aios/scheduler/common.py`
- `python/aios/scheduler/prefill.py`
- `python/aios/scheduler/decode.py`
- `python/aios/scheduler/scheduler.py`

重点理解：

- 请求状态如何从 waiting 进入 running
- prefill batch 如何构造
- decode batch 如何构造
- continuous batching 如何让新请求插入执行流
- 每次 forward 后如何采样、追加 token、释放完成请求

这部分是当前源码里最容易绕的地方，建议配合 `debug_scheduler=True` 运行小 batch，再对照状态变化读。

### 7. Fused layers

先读文档：

- `resources/lesson-9-fused-layers/README_CN.md`

再回看源码：

- `python/aios/layers/linear.py`
- `python/aios/layers/norm.py`
- `python/aios/layers/activation.py`
- `python/aios/models/weight.py`
- `python/aios/models/qwen3.py`

重点理解：

- Q/K/V 为什么可以打包成一个 projection
- gate/up 为什么可以打包成一个 projection
- fused add rmsnorm 合并了哪些操作
- checkpoint 原始权重名如何映射到 fused 模块权重名

### 8. CUDA graphs

先读文档：

- `resources/lesson-10-cuda-graphs/README_CN.md`

再读源码：

- `python/aios/engine/graph.py`
- `python/aios/engine/engine.py`
- `python/aios/attention/base.py`
- `python/aios/attention/fi.py`
- `python/aios/scheduler/scheduler.py`
- `python/aios/llm/llm.py`

重点理解：

- 为什么只 capture decode，不 capture prefill
- static tensor 地址为什么重要
- 动态 batch size 如何映射到固定 CUDA graph batch size
- scheduler 如何决定用 eager forward 还是 graph replay

## 四、建议的三遍阅读法

### 第一遍：只建立地图

目标：知道每个目录负责什么。

阅读顺序：

```text
README.md
resources/lesson-0-introduction/README_CN.md
resources/lesson-1-llm-basics/README_CN.md
resources/lesson-2-run-qwen3/README.md
resources/lesson-3-refactor-to-package/README.md
python/aios/layers/base.py
python/aios/models/qwen3.py
python/aios/llm/llm.py
```

这一遍不要纠结 FlashInfer、CUDA graph、page table 细节。

### 第二遍：按一次请求的生命周期追代码

目标：理解一条 prompt 如何变成输出 token。

推荐调用链：

```text
python/aios/__main__.py
-> python/aios/llm/llm.py: LLM.generate
-> python/aios/scheduler/scheduler.py
-> python/aios/scheduler/prefill.py
-> python/aios/engine/engine.py
-> python/aios/models/qwen3.py
-> python/aios/attention/fi.py
-> python/aios/engine/sample.py
-> python/aios/scheduler/decode.py
```

这一遍重点看数据结构的变化：`Req`、`Batch`、`positions`、`input_ids`、`out_loc`、page table。

### 第三遍：按优化主题回看

目标：理解为什么每一课都能提升性能。

主题顺序：

```text
无 KV cache baseline
-> dynamic KV cache
-> paged KV cache
-> static batching
-> continuous batching
-> flat varlen prefill
-> fused layers
-> CUDA graphs
```

每个主题都先看对应 Markdown，再看对应源码。

## 五、最重要的文件清单

### 必须读

- `README.md`
- `resources/lesson-1-llm-basics/README_CN.md`
- `resources/lesson-2-run-qwen3/README.md`
- `resources/lesson-2-run-qwen3/run_qwen3.py`
- `resources/lesson-3-refactor-to-package/README.md`
- `python/aios/layers/base.py`
- `python/aios/models/qwen3.py`
- `python/aios/models/weight.py`
- `python/aios/core.py`
- `python/aios/llm/llm.py`

### 理解 KV cache 和调度时必须读

- `resources/lesson-4-kv-cache/README_CN.md`
- `resources/lesson-5-paged-kv-cache/README_CN.md`
- `resources/lesson-6-static-batching/TECHNICAL_CN.md`
- `resources/lesson-7-continuous-batching/TECHNICAL_CN.md`
- `resources/lesson-8-flat-varlen-prefill/README_CN.md`
- `python/aios/kvcache/base.py`
- `python/aios/kvcache/mha_pool.py`
- `python/aios/scheduler/table.py`
- `python/aios/scheduler/cache.py`
- `python/aios/scheduler/prefill.py`
- `python/aios/scheduler/decode.py`
- `python/aios/scheduler/scheduler.py`

### 理解性能优化时必须读

- `resources/lesson-9-fused-layers/README_CN.md`
- `resources/lesson-10-cuda-graphs/README_CN.md`
- `python/aios/layers/linear.py`
- `python/aios/layers/norm.py`
- `python/aios/models/weight.py`
- `python/aios/attention/base.py`
- `python/aios/attention/fi.py`
- `python/aios/engine/graph.py`

## 六、阅读时的几个提醒

1. 当前源码不是 Lesson 3 的纯净版本。Lesson 3 文档里的很多代码片段是“包化后的基础版”，当前 `python/aios/` 已经继续加入后续课程功能。
2. 遇到 `get_global_ctx()` 时，不要把它当作 Lesson 3 的核心。它主要服务后续 batching、attention backend 和 CUDA graph。
3. 遇到 `LinearQKVMerged`、`LinearColParallelMerged`、`RMSNormFused` 时，先知道这是 Lesson 9 的融合优化，不影响你理解原始 Qwen3 结构。
4. 遇到 page table、`out_loc`、`qo_indptr`、`kv_indptr` 时，先去读 Lesson 5、Lesson 8 文档，再回来看源码。
5. 阅读 scheduler 时一定要配合 `Req` 和 `Batch` 看，否则容易只看到控制流，看不到状态机。

## 七、如果只想先掌握 Lesson 3

最短路线是：

```text
resources/lesson-2-run-qwen3/run_qwen3.py
-> resources/lesson-3-refactor-to-package/README.md
-> python/aios/layers/base.py
-> python/aios/layers/linear.py
-> python/aios/layers/embedding.py
-> python/aios/layers/rotary.py
-> python/aios/models/config.py
-> python/aios/models/qwen3.py
-> python/aios/models/weight.py
-> python/aios/engine/sample.py
-> python/aios/llm/llm.py
```

这一轮只回答一个问题：

```text
run_qwen3.py 里的每一段代码，被移动到了 aios 包里的哪个模块？
```

能回答这个问题，就说明 Lesson 3 的主线已经掌握。

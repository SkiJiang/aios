# `python/aios/__main__.py` 源码解释报告

## 文件定位

这个文件提供命令行入口，使用户可以通过下面的方式运行 AIOS：

```bash
python -m aios --model /path/to/model --prompt "Who are you?"
```

它不实现推理逻辑，只负责解析命令行参数、创建 `LLM`、调用 `generate()` 并打印结果。

## 顶部说明

文件开头的 docstring 给出两个用法示例：

```text
python -m aios --model /path/to/Qwen3-0.6B --prompt "Who are you?"
python -m aios --model /path/to/Qwen3-0.6B --prompt "Hi" --prompt "Hello" --max-running-reqs 2
```

第二个示例说明 `--prompt` 可以重复传入，从而构造 batch。

## 导入

```python
import argparse
import time

from aios import LLM, SamplingParams
```

`argparse` 用于命令行参数解析。

`time.perf_counter()` 用于统计模型加载耗时。

`LLM` 是推理入口，`SamplingParams` 是生成配置。

## `main()`

`main()` 是 CLI 的主函数。

它先创建参数解析器：

```python
parser = argparse.ArgumentParser(description="AIOS LLM Inference Engine")
```

支持的参数包括：

- `--model`：模型目录或 HuggingFace 模型名，必填。
- `--prompt`：输入 prompt，可重复传入。
- `--max-tokens`：最大生成 token 数。
- `--temperature`：采样温度，0 表示贪心。
- `--top-k`：top-k 采样，-1 表示关闭。
- `--device`：CUDA 设备。
- `--max-running-reqs`：最多并发运行请求数。

如果没有传入 prompt，默认使用：

```python
["Who are you?"]
```

## 模型加载

```python
llm = LLM(args.model, device=args.device)
```

这一步会触发 `LLM.__init__`，内部会下载或解析模型路径、加载 tokenizer、创建 engine、加载权重、分配 KV cache。

加载耗时通过：

```python
t_load = time.perf_counter() - t0
```

输出到终端。

## 采样参数

```python
sampling_params = SamplingParams(
    temperature=args.temperature,
    top_k=args.top_k,
    max_tokens=args.max_tokens,
)
```

CLI 只暴露了部分采样参数，未暴露 `top_p` 和 `ignore_eos`，它们使用 `SamplingParams` 默认值。

## 调用生成

```python
results = llm.generate(prompts, sampling_params, max_running_reqs=args.max_running_reqs)
```

这里把 prompt 列表、采样参数和并发上限交给 `LLM.generate()`。

返回的 `results` 是 dict 列表，每个元素包含生成 token 和解码文本。

## 输出结果

如果 batch 中有多个 prompt，会打印 prompt 标题：

```python
=== prompt 0: '...' ===
```

然后打印生成文本：

```python
print(r["text"])
```

## 入口保护

```python
if __name__ == "__main__":
    main()
```

这保证只有以脚本方式执行或 `python -m aios` 时才调用 `main()`。

## 总结

`__main__.py` 是 AIOS 的命令行包装层。它把命令行参数转成 `LLM` 和 `SamplingParams` 调用，不直接参与模型 forward、调度、KV cache 或 attention 计算。

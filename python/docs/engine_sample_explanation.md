# `python/aios/engine/sample.py` 源码解释报告

## 文件定位

`sample.py` 定义 `Sampler`，负责根据模型输出 logits 选择下一个 token。

它是 engine forward 后的最后一步。

## `Sampler`

```python
@dataclass
class Sampler:
    sampling_params: SamplingParams
```

每个 sampler 绑定一个请求的采样参数。

`Engine.forward_batch()` 会为 batch 中每个请求分别创建 `Sampler`：

```python
token = Sampler(req.sampling_params).sample(logits[i : i + 1])
```

## `sample`

输入：

```text
logits.shape = [batch, vocab_size]
```

当前调用中通常是单请求 slice：

```text
[1, vocab_size]
```

## 贪心路径

```python
if self.sampling_params.is_greedy:
    return logits.argmax(dim=-1, keepdim=True)
```

如果采样参数表示 greedy，直接选择 logits 最大的 token。

返回形状：

```text
[batch, 1]
```

## 非贪心路径

先除以 temperature：

```python
logits = logits / self.sampling_params.temperature
```

如果启用 top-k：

```python
topk_vals = torch.topk(logits, min(top_k, vocab_size)).values
logits = logits.masked_fill(logits < topk_vals[..., -1:], -inf)
```

只保留 top-k 范围内的 token，其余 logits 设为负无穷。

然后 softmax：

```python
probs = F.softmax(logits, dim=-1)
```

最后多项式采样：

```python
torch.multinomial(probs, num_samples=1)
```

## 注意点

`SamplingParams` 中有 `top_p` 字段，但当前 `sample.py` 没有实现 top-p 过滤。`top_p` 只参与 `is_greedy` 判断。

## 总结

`Sampler` 实现 greedy 和 temperature/top-k 采样，是 logits 到 next token 的转换层。它不关心模型结构，也不关心调度，只根据每个请求的 `SamplingParams` 工作。

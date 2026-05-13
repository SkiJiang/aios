from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from aios.layers import (
    BaseOP,
    Embedding,
    Linear,
    LMHead,
    OPList,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    silu_and_mul,
)

from .base import BaseLLMModel

if TYPE_CHECKING:
    from aios.core import Batch
    from .config import ModelConfig
    from aios.kvcache import MHAKVCache


class Qwen3Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self._scale = config.head_dim ** -0.5
        self._layer_idx = layer_idx

        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.v_proj = Linear(config.hidden_size, self.num_kv_heads * self.head_dim)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        paged_kv_cache: MHAKVCache,
        batch: Batch,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj.forward(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj.forward(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj.forward(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        return self._batched_paged_attention(
            q, k, v, paged_kv_cache, batch, bsz, seq_len
        )

    def _batched_paged_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        batch: Batch,
        bsz: int,
        seq_len: int,
    ) -> torch.Tensor:
        device = q.device
        dtype = q.dtype
        assert batch.page_table is not None
        page_table = batch.page_table

        # Store KV for each request
        for i, req in enumerate(batch.reqs):
            out_loc_i = batch.out_loc[i, :seq_len]
            k_i = k[i].transpose(0, 1)  # (seq_len, num_kv_heads, head_dim)
            v_i = v[i].transpose(0, 1)
            paged_kv_cache.store_kv(k_i, v_i, out_loc_i, self._layer_idx)

        # Retrieve full KV and compute attention
        if batch.is_prefill:
            # Prefill: all reqs share the same kv_len in this batch (bsz=1 in lesson 7).
            kv_len = batch.reqs[0].cached_len + seq_len
            k_list, v_list = [], []
            for req in batch.reqs:
                all_locs = page_table[req.table_idx, :kv_len]
                k_i = paged_kv_cache.k_cache(self._layer_idx)[all_locs, :, 0, :]
                v_i = paged_kv_cache.v_cache(self._layer_idx)[all_locs, :, 0, :]
                k_list.append(k_i.transpose(0, 1))  # (heads, kv_len, dim)
                v_list.append(v_i.transpose(0, 1))
            k_full = torch.stack(k_list)  # (B, heads, kv_len, dim)
            v_full = torch.stack(v_list)

            q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
            k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
            causal_mask = torch.where(
                k_pos > q_pos,
                torch.tensor(float("-inf"), device=device, dtype=dtype),
                torch.tensor(0.0, device=device, dtype=dtype),
            ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, kv_len)
        else:
            # Decode: variable kv_len per request, pad to max
            max_kv_len = max(r.cached_len + 1 for r in batch.reqs)
            k_list, v_list = [], []
            for req in batch.reqs:
                kv_len_i = req.cached_len + 1
                all_locs = page_table[req.table_idx, :kv_len_i]
                k_i = paged_kv_cache.k_cache(self._layer_idx)[all_locs, :, 0, :]
                v_i = paged_kv_cache.v_cache(self._layer_idx)[all_locs, :, 0, :]
                if kv_len_i < max_kv_len:
                    pad_len = max_kv_len - kv_len_i
                    k_i = F.pad(k_i, (0, 0, 0, 0, 0, pad_len))
                    v_i = F.pad(v_i, (0, 0, 0, 0, 0, pad_len))
                k_list.append(k_i.transpose(0, 1))
                v_list.append(v_i.transpose(0, 1))
            k_full = torch.stack(k_list)
            v_full = torch.stack(v_list)

            causal_mask = torch.full(
                (bsz, 1, 1, max_kv_len), float("-inf"), device=device, dtype=dtype
            )
            for i, req in enumerate(batch.reqs):
                causal_mask[i, 0, 0, : req.cached_len + 1] = 0.0

        k_full = repeat_kv(k_full, self.num_kv_groups)
        v_full = repeat_kv(v_full, self.num_kv_groups)

        attn_weights = torch.matmul(q, k_full.transpose(-2, -1)) * self._scale
        attn_weights = attn_weights + causal_mask
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_probs, v_full)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj.forward(attn_output)


class Qwen3MLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size)
        match config.hidden_act:
            case "silu":
                self._act_fn = silu_and_mul
            case act_fn:
                raise ValueError(f"Unsupported activation: {act_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(self._act_fn(self.gate_proj.forward(x), self.up_proj.forward(x)))


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        paged_kv_cache: MHAKVCache,
        batch: Batch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states = self.self_attn.forward(
            hidden_states, position_embeddings, paged_kv_cache, batch
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([Qwen3DecoderLayer(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        batch: Batch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens.forward(input_ids)
        position_embeddings = self._rotary_emb.forward(batch.positions)

        for layer in self.layers.op_list:
            hidden_states = layer.forward(
                hidden_states, position_embeddings, paged_kv_cache, batch
            )
        return self.norm.forward(hidden_states)


class Qwen3ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = LMHead(
            config.vocab_size,
            config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        paged_kv_cache: MHAKVCache,
        batch: Batch,
    ) -> torch.Tensor:
        hidden_states = self.model.forward(input_ids, paged_kv_cache, batch)
        return self.lm_head.forward(hidden_states)

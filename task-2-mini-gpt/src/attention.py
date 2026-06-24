from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[dict[str, Tensor]] = None,
        return_cache: bool = False,
    ):
        batch_size, seq_len, _ = x.shape
        device = x.device

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        past_len = 0 if kv_cache is None else kv_cache["k"].size(-2)
        cos, sin = build_rope_cache(
            seq_len=past_len + seq_len,
            head_dim=self.head_dim,
            device=device,
        )
        q = apply_rope(q, cos, sin, offset=past_len)
        k = apply_rope(k, cos, sin, offset=past_len)

        if kv_cache is not None:
            k = torch.cat([kv_cache["k"], k], dim=-2)
            v = torch.cat([kv_cache["v"], v], dim=-2)

        attn_mask = None
        if kv_cache is None:
            attn_mask = torch.triu(
                torch.ones(seq_len, k.size(-2), device=device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            total_len = k.size(-2)
            positions = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(-1)
            key_positions = torch.arange(total_len, device=device).unsqueeze(0)
            attn_mask = key_positions > positions

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = self.out_proj(self._merge_heads(out))

        new_cache = {"k": k, "v": v}
        if return_cache:
            return out, new_cache
        return out

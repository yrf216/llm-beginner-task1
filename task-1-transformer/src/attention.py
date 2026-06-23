import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _broadcast_mask(mask: Tensor, target_ndim: int) -> Tensor:
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(0)
    return mask.to(dtype=torch.bool)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_attention: bool = False,
):
    """Manual scaled dot-product attention.

    Args:
        Q, K, V: shape (B, H, T, D)
        mask: broadcastable to (B, H, T, S), True means masked.
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = _broadcast_mask(mask, scores.dim())
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attention = torch.softmax(scores, dim=-1)
    attention = torch.nan_to_num(attention, nan=0.0)

    if dropout_p > 0.0:
        attention = F.dropout(attention, p=dropout_p, training=training)

    output = torch.matmul(attention, V)
    if return_attention:
        return output, attention
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        attn_output, attention = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout_p=self.dropout,
            training=self.training,
            return_attention=True,
        )
        output = self.out_proj(self._merge_heads(attn_output))
        if return_attention:
            return output, attention
        return output

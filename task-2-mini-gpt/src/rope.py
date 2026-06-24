from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, device: torch.device, base: int = 10000):
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dims = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (dims / head_dim))
    angles = torch.outer(positions, inv_freq)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    seq_len = x.size(-2)
    cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

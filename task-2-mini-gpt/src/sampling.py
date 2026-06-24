from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= top_p
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        next_sorted = torch.multinomial(filtered, num_samples=1)
        return sorted_idx.gather(-1, next_sorted)

    return torch.multinomial(probs, num_samples=1)

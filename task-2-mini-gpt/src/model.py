from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .attention import CausalSelfAttention
from .sampling import sample_next_token
from .tokenizer import BPETokenizer


@dataclass
class MiniGPTConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_hidden_dim: int = 384
    dropout: float = 0.1


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(
            d_model=config.d_model,
            hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor, kv_cache=None, return_cache: bool = False):
        attn_out, new_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, return_cache=True)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        if return_cache:
            return x, new_cache
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.max_seq_len
        self.max_seq_len = config.max_seq_len

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, ids: Tensor, kv_cache=None, return_cache: bool = False):
        x = self.dropout(self.token_embedding(ids))

        new_caches = []
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for block, layer_cache in zip(self.blocks, kv_cache):
            x, layer_new_cache = block(x, kv_cache=layer_cache, return_cache=True)
            new_caches.append(layer_new_cache)

        logits = self.lm_head(self.final_norm(x))
        if return_cache:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int,
        top_p: float = 0.9,
        temperature: float = 1.0,
    ) -> list[int]:
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        kv_cache = None
        generated = ids

        for _ in range(max_new_tokens):
            if kv_cache is None:
                logits, kv_cache = self(generated[:, -self.max_seq_len :], return_cache=True)
            else:
                logits, kv_cache = self(generated[:, -1:], kv_cache=kv_cache, return_cache=True)
            next_id = sample_next_token(logits[:, -1, :], top_p=top_p, temperature=temperature)
            generated = torch.cat([generated, next_id], dim=1)
            if generated.size(1) > self.max_seq_len:
                generated = generated[:, -self.max_seq_len :]
                kv_cache = None
        return generated[0].tolist()


def save_checkpoint(
    ckpt_path: str | Path,
    model: MiniGPT,
    config: MiniGPTConfig,
    tokenizer_path: str | Path,
    metrics: dict,
) -> None:
    payload = {
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "tokenizer_path": str(tokenizer_path),
        "metrics": metrics,
    }
    torch.save(payload, ckpt_path)


def load_for_eval(ckpt_path: str):
    payload = torch.load(ckpt_path, map_location="cpu")
    config = MiniGPTConfig(**payload["config"])
    model = MiniGPT(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    tokenizer_path = Path(payload.get("tokenizer_path", Path(ckpt_path).with_name("tokenizer.json")))
    if not tokenizer_path.is_absolute():
        tokenizer_path = Path(ckpt_path).resolve().parent / tokenizer_path.name
    tokenizer = BPETokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

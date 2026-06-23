from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

from .block import TransformerBlock
from .tokenizer import SimpleCharTokenizer


@dataclass
class TransformerConfig:
    vocab_size: int
    max_length: int
    num_classes: int = 2
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 3
    ffn_hidden_dim: int = 256
    dropout: float = 0.15
    pad_id: int = 0


class TransformerClassifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    hidden_dim=config.ffn_hidden_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_classes),
        )

    def forward(self, input_ids: Tensor, return_attention: bool = False):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        padding_mask = input_ids.eq(self.config.pad_id).unsqueeze(1).unsqueeze(2)
        attentions = []
        for block in self.blocks:
            if return_attention:
                x, attention = block(x, mask=padding_mask, return_attention=True)
                attentions.append(attention)
            else:
                x = block(x, mask=padding_mask)

        x = self.final_norm(x)

        non_pad = input_ids.ne(self.config.pad_id)
        content_mask = non_pad.clone()
        content_mask[:, 0] = False
        content_mask_f = content_mask.unsqueeze(-1).float()
        mean_repr = (x * content_mask_f).sum(dim=1) / content_mask_f.sum(dim=1).clamp(min=1.0)
        cls_repr = x[:, 0]
        pooled = torch.cat([cls_repr, mean_repr], dim=-1)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attentions
        return logits


def save_checkpoint(
    ckpt_path: str | Path,
    model: TransformerClassifier,
    tokenizer: SimpleCharTokenizer,
    metrics: dict,
) -> None:
    ckpt = {
        "config": asdict(model.config),
        "state_dict": model.state_dict(),
        "vocab": tokenizer.token_to_id,
        "metrics": metrics,
    }
    torch.save(ckpt, ckpt_path)


def load_for_eval(ckpt_path: str) -> tuple[TransformerClassifier, Callable[[str], Tensor]]:
    payload = torch.load(ckpt_path, map_location="cpu")
    config = TransformerConfig(**payload["config"])
    model = TransformerClassifier(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    tokenizer = SimpleCharTokenizer(payload["vocab"])

    def tokenize_fn(text: str) -> Tensor:
        return tokenizer.encode(text, max_length=config.max_length)

    return model, tokenize_fn

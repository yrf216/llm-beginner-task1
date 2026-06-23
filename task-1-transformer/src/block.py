from typing import Optional

from torch import Tensor, nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=hidden_dim, dropout=dropout)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ):
        attn_input = self.attn_norm(x)
        attn_output, attention = self.attn(attn_input, mask=mask, return_attention=True)
        x = x + self.attn_dropout(attn_output)

        ffn_input = self.ffn_norm(x)
        x = x + self.ffn_dropout(self.ffn(ffn_input))

        if return_attention:
            return x, attention
        return x

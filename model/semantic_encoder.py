

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):


    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, out_dim))
        self.attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:

        B = h.size(0)
        v = self.proj(h)                            # [B, T, D_out]
        q = self.query.expand(B, -1, -1)            # [B, 1, D_out]
        out, _ = self.attn(q, v, v, need_weights=False)  # [B, 1, D_out]
        return self.norm(out.squeeze(1))            # [B, D_out]


class SemanticEncoder(nn.Module):


    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.attn_pool = AttentionPooling(in_dim, out_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.normalize = normalize
        self.out_dim = out_dim

    def forward(self, h_shared: torch.Tensor) -> torch.Tensor:
        """
        h_shared: [B, T, D]
        returns: z_s [B, Ds]
        """
        pooled = self.attn_pool(h_shared)       # [B, Ds]
        z = self.norm(self.mlp(pooled) + pooled)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z

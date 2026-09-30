
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── TCN Block ─────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNResidualBlock(nn.Module):

    def __init__(self, channels: int, kernel: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


# ─── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """标准 Pre-LN Transformer block"""

    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


# ─── Positional Encoding ──────────────────────────────────────────────────────

class SinPosEncoding(nn.Module):

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ─── Shared Encoder ───────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 6,
        stem_channels: int = 64,
        tcn_channels: int = 128,
        tcn_layers: int = 3,
        tcn_kernel: int = 5,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── Conv1d stem ──────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.GELU(),
            nn.Conv1d(stem_channels, tcn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(tcn_channels),
            nn.GELU(),
        )

        # ── TCN residual blocks ──────────────────────────────────────────────
        self.tcn_blocks = nn.ModuleList([
            TCNResidualBlock(
                tcn_channels,
                kernel=tcn_kernel,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(tcn_layers)
        ])

        # ── Projection to transformer dim ────────────────────────────────────
        self.proj = nn.Linear(tcn_channels, transformer_dim) \
            if tcn_channels != transformer_dim else nn.Identity()

        # ── Transformer ──────────────────────────────────────────────────────
        self.pos_enc = SinPosEncoding(transformer_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(transformer_dim, transformer_heads, dropout=dropout)
            for _ in range(transformer_layers)
        ])

        self.out_norm = nn.LayerNorm(transformer_dim)
        self.out_dim = transformer_dim

    def forward(self, x: torch.Tensor) -> dict:
        # Conv stem + TCN  (channel-first)
        h = self.stem(x)              # [B, tcn_ch, T]
        for block in self.tcn_blocks:
            h = block(h)              # [B, tcn_ch, T]

        # Transpose for transformer  (batch-first)
        h = h.permute(0, 2, 1)       # [B, T, tcn_ch]
        h = self.proj(h)              # [B, T, D]
        h = self.pos_enc(h)

        for block in self.transformer_blocks:
            h = block(h)              # [B, T, D]

        h = self.out_norm(h)          # [B, T, D]
        h_pool = h.mean(dim=1)        # [B, D]

        return {"h_shared": h, "h_pool": h_pool}

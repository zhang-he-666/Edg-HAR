import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.base_dataset import SAMPLING_RATE_BINS


class MetaEmbedding(nn.Module):

    def __init__(
        self,
        num_datasets: int = 10,
        num_sr_bins: int = 4,
        num_positions: int = 8,
        num_sensor_types: int = 4,
        d_embed: int = 32,
    ):
        super().__init__()
        self.emb_dataset = nn.Embedding(num_datasets, d_embed)
        self.emb_sr = nn.Embedding(num_sr_bins, d_embed // 2)
        self.emb_pos = nn.Embedding(num_positions, d_embed // 2)
        self.emb_sensor = nn.Embedding(num_sensor_types, d_embed // 2)

        out_dim = d_embed + 3 * (d_embed // 2)
        self.proj = nn.Linear(out_dim, d_embed)
        self.out_dim = d_embed

    def forward(self, meta: torch.Tensor) -> torch.Tensor:

        e_ds = self.emb_dataset(meta[:, 0])
        e_sr = self.emb_sr(meta[:, 1])
        e_pos = self.emb_pos(meta[:, 2])
        e_sen = self.emb_sensor(meta[:, 3])
        combined = torch.cat([e_ds, e_sr, e_pos, e_sen], dim=-1)
        return self.proj(combined)


class AcquisitionEncoder(nn.Module):


    def __init__(
        self,
        in_dim: int = 128,
        out_dim: int = 64,
        hidden_dim: int = 128,
        d_meta: int = 32,
        num_datasets: int = 10,
        num_sr_bins: int = 4,
        num_positions: int = 8,
        num_sensor_types: int = 4,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()

        self.meta_emb = MetaEmbedding(
            num_datasets=num_datasets,
            num_sr_bins=num_sr_bins,
            num_positions=num_positions,
            num_sensor_types=num_sensor_types,
            d_embed=d_meta,
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_dim + d_meta, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.normalize = normalize
        self.out_dim = out_dim

    def forward(
        self,
        h_pool: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        meta_feat = self.meta_emb(meta)                    # [B, d_meta]
        feat = torch.cat([h_pool, meta_feat], dim=-1)      # [B, D + d_meta]
        z = self.out_norm(self.mlp(feat))
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z

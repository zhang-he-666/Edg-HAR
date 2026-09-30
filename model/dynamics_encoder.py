import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrFeature(nn.Module):

    def __init__(self, max_lag: int = 32):
        super().__init__()
        self.max_lag = max_lag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_centered = x - x.mean(dim=-1, keepdim=True)
        var = (x_centered ** 2).mean(dim=-1, keepdim=True) + 1e-6
        if T <= self.max_lag:
            return torch.zeros(B, C, self.max_lag, device=x.device)
        unfolded = x_centered.unfold(2, self.max_lag, 1)           # [B, C, n, max_lag] where n = T-max_lag+1
        n = unfolded.size(2)
        base = x_centered[:, :, :n]                                # [B, C, n]
        corr = (base.unsqueeze(-1) * unfolded).mean(dim=2)         # [B, C, max_lag]
        corr = corr / var                                           # normalize by variance
        return corr  # [B, C, max_lag]


class SpectralBandFeature(nn.Module):
    def __init__(self, n_bands: int = 8):
        super().__init__()
        self.n_bands = n_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        X = torch.fft.rfft(x, dim=-1)          # [B, C, F]
        power = X.real ** 2 + X.imag ** 2       # [B, C, F]
        F_bins = power.size(-1)
        band_size = max(1, F_bins // self.n_bands)
        bands = []
        for i in range(self.n_bands):
            s = i * band_size
            e = min(s + band_size, F_bins)
            bands.append(power[:, :, s:e].mean(dim=-1))  # [B, C]
        feat = torch.stack(bands, dim=-1)       # [B, C, n_bands]
        return feat.reshape(B, -1)              # [B, C * n_bands]



class DynamicsEncoder(nn.Module):


    def __init__(
        self,
        in_dim: int = 128,       
        in_channels: int = 6,       
        out_dim: int = 64,
        hidden_dim: int = 128,
        use_autocorr: bool = True,
        use_spectral: bool = True,
        autocorr_lags: int = 32,
        spectral_bands: int = 8,
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.use_autocorr = use_autocorr
        self.use_spectral = use_spectral
        self.in_channels = in_channels

        # ── Temporal attention pooling on h_shared ───────────────────────────
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # -> [B, in_dim, 1]

        feat_dim = in_dim
        if use_autocorr:
            self.autocorr = AutoCorrFeature(max_lag=autocorr_lags)
            feat_dim += in_channels * autocorr_lags
        if use_spectral:
            self.spectral = SpectralBandFeature(n_bands=spectral_bands)
            feat_dim += in_channels * spectral_bands

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
  
        self.normalize = False
        self.scale_max = 2.0  
        self.out_dim = out_dim

    def forward(
        self,
        h_shared: torch.Tensor,
        x_time: torch.Tensor = None,
        x_fft: torch.Tensor = None,
    ) -> torch.Tensor:

        B = h_shared.size(0)

        h = h_shared.permute(0, 2, 1)              # [B, D, T]
        h = self.temporal_conv(h)                   # [B, D, T]
        h_pooled = self.pool(h).squeeze(-1)         # [B, D]

        feats = [h_pooled]

        if self.use_autocorr:
            if x_time is not None:
                ac = self.autocorr(x_time)              # [B, C, max_lag]
                feats.append(ac.reshape(B, -1))         # [B, C*max_lag]
            else:
                ac_dim = self.in_channels * self.autocorr.max_lag
                feats.append(torch.zeros(B, ac_dim, device=h_shared.device))

        if self.use_spectral:
            if x_fft is not None:
                F_bins = x_fft.size(-1)
                n_bands = self.spectral.n_bands
                band_size = max(1, F_bins // n_bands)
                bands = []
                for i in range(n_bands):
                    s = i * band_size
                    e = min(s + band_size, F_bins)
                    bands.append(x_fft[:, :, s:e].mean(dim=-1))  # [B, C]
                sp = torch.stack(bands, dim=-1).reshape(B, -1)   # [B, C*n_bands]
                feats.append(sp)
            elif x_time is not None:
                sp = self.spectral(x_time)          # [B, C*n_bands]
                feats.append(sp)
            else:
                sp_dim = self.in_channels * self.spectral.n_bands
                feats.append(torch.zeros(B, sp_dim, device=h_shared.device))

        feat = torch.cat(feats, dim=-1)             # [B, feat_dim]
        z = self.out_norm(self.mlp(feat))
        if self.normalize:
            z = F.normalize(z, dim=-1)
        else:
            z = self.scale_max * torch.tanh(z / self.scale_max)
        return z

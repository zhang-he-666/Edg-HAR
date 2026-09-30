

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorDecoder(nn.Module):


    def __init__(
        self,
        dim_s: int = 128,
        dim_d: int = 64,
        dim_a: int = 64,
        hidden_dim: int = 256,
        out_dim: int = 128,  
        num_layers: int = 2,
    ):
        super().__init__()
        in_dim = dim_s + dim_d + dim_a

        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(
        self,
        z_s: torch.Tensor,
        z_d: torch.Tensor,
        z_a: torch.Tensor,
    ) -> torch.Tensor:

        z = torch.cat([z_s, z_d, z_a], dim=-1)
        return self.net(z)


class SwapReconstructor(nn.Module):


    def __init__(self, decoder: FactorDecoder):
        super().__init__()
        self.decoder = decoder

    def swap_acquisition(
        self,
        z_s_i: torch.Tensor,
        z_d_i: torch.Tensor,
        z_a_j: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(z_s_i, z_d_i, z_a_j)

    def swap_dynamics(
        self,
        z_s_i: torch.Tensor,
        z_d_j: torch.Tensor,
        z_a_i: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(z_s_i, z_d_j, z_a_i)

    def forward(
        self,
        z_s: torch.Tensor,
        z_d: torch.Tensor,
        z_a: torch.Tensor,
        same_class_idx: torch.Tensor,    
        swap_type: str = "acquisition",  
    ) -> dict:

        j = same_class_idx
        results = {}

        if swap_type in ("acquisition", "both"):
            x_swap_a = self.swap_acquisition(z_s, z_d, z_a[j])
            results["swap_a"] = x_swap_a   # [B, out_dim]

        if swap_type in ("dynamics", "both"):
            x_swap_d = self.swap_dynamics(z_s, z_d[j], z_a)
            results["swap_d"] = x_swap_d   # [B, out_dim]

        return results


def build_same_class_pairs(labels: torch.Tensor) -> torch.Tensor:
    B = labels.size(0)
    idx = torch.arange(B, device=labels.device)
    for c in labels.unique():
        same = (labels == c).nonzero(as_tuple=True)[0]
        if len(same) < 2:
            continue
        perm = same[torch.randperm(len(same), device=labels.device)]
        idx[same] = perm

        self_mask = (idx[same] == same)
        if self_mask.any():
            self_pos = self_mask.nonzero(as_tuple=True)[0]
            for p in self_pos:
                neighbor = (p + 1) % len(same)
                idx[same[p]] = same[neighbor]
    return idx

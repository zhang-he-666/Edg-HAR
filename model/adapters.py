import torch
import torch.nn as nn


class MLPAdapter(nn.Module):

    def __init__(self, in_dim: int, bottleneck_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, in_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_dim)
        # Small-scale init: up initialized with small values (not zero) so
        # gradients flow through down/up from the first step.
        # Scale 0.01 keeps initial output delta small (~identity) but active.
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.drop(self.act(self.down(z))))
        return self.norm(z + delta)


class AcquisitionAdapter(nn.Module):

    def __init__(self, in_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.adapter = MLPAdapter(in_dim, bottleneck_dim)
        self._enabled = True

    def enable(self):
        self._enabled = True
        for p in self.parameters():
            p.requires_grad_(True)

    def disable(self):
        self._enabled = False
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        if self._enabled:
            return self.adapter(z_a)
        return z_a


def collect_norm_params(model: nn.Module) -> list:

    params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
    return params

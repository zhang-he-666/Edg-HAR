import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ─── Gradient Reversal ────────────────────────────────────────────────────────

class GRLFunction(Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):

    def __init__(self, max_alpha: float = 1.0):
        super().__init__()
        self.max_alpha = max_alpha
        self._step = 0
        self._schedule_steps = 5000

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.max_alpha * min(1.0, self._step / self._schedule_steps)
        return GRLFunction.apply(x, alpha)

    def step(self):
        self._step += 1

    def set_schedule_steps(self, steps: int):
        self._schedule_steps = steps


# ─── Classification Heads ─────────────────────────────────────────────────────

class ClassificationHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.net(z_s)


class AuxClassificationHead(nn.Module):

    def __init__(
        self,
        dim_s: int,
        dim_d: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_s + dim_d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        z_d_sg = z_d.detach()
        return self.net(torch.cat([z_s, z_d_sg], dim=-1))


# ─── Prototype Head ───────────────────────────────────────────────────────────

class ProtoHead(nn.Module):

    def __init__(self, in_dim: int, num_classes: int, temperature: float = 0.1):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim))
        nn.init.orthogonal_(self.prototypes)  
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(self, z_s: torch.Tensor) -> dict:

        P = F.normalize(self.prototypes, dim=-1)   # [C, Ds]
        sim = z_s @ P.T                             # [B, C]
        return {"sim": sim, "logits": sim / self.temperature}

    def get_prototype_distance_matrix(self) -> torch.Tensor:
        P = F.normalize(self.prototypes, dim=-1)
        diff = P.unsqueeze(0) - P.unsqueeze(1)     # [C, C, D]
        return (diff ** 2).sum(-1).clamp(min=1e-8).sqrt()           # [C, C]


# ─── Domain Heads ─────────────────────────────────────────────────────────────

class DomainHeadOnSemantic(nn.Module):

    def __init__(self, in_dim: int, num_domains: int, hidden_dim: int = 64, max_alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(max_alpha)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z_s: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(z_s))

    def step_grl(self):
        self.grl.step()


class DomainHeadOnAcquisition(nn.Module):

    def __init__(self, in_dim: int, num_domains: int, hidden_dim: int = 64):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_a)


class DomainHeadOnDynamics(nn.Module):

    def __init__(self, in_dim: int, num_domains: int, hidden_dim: int = 64, max_alpha: float = 1.0):
        super().__init__()
        self.grl = GradientReversal(max_alpha)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, z_d: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(z_d))

    def step_grl(self):
        self.grl.step()

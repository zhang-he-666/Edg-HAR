import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .shared_encoder import SharedEncoder
from .semantic_encoder import SemanticEncoder
from .dynamics_encoder import DynamicsEncoder
from .acquisition_encoder import AcquisitionEncoder
from .heads import (
    ClassificationHead,
    AuxClassificationHead,
    ProtoHead,
    DomainHeadOnSemantic,
    DomainHeadOnAcquisition,
    DomainHeadOnDynamics,
)
from .decoder import FactorDecoder, SwapReconstructor, build_same_class_pairs
from .adapters import AcquisitionAdapter


class FactorizedHARModel(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        c = cfg

        # ── 共享编码器 ────────────────────────────────────────────────────────
        enc_cfg = c["model"]["shared_encoder"]
        self.shared_encoder = SharedEncoder(
            in_channels=enc_cfg["in_channels"],
            stem_channels=enc_cfg["stem_channels"],
            tcn_channels=enc_cfg["tcn_channels"],
            tcn_layers=enc_cfg["tcn_layers"],
            tcn_kernel=enc_cfg["tcn_kernel"],
            transformer_dim=enc_cfg["transformer_dim"],
            transformer_heads=enc_cfg["transformer_heads"],
            transformer_layers=enc_cfg["transformer_layers"],
            dropout=enc_cfg["dropout"],
        )
        D = enc_cfg["transformer_dim"]

        dim_s = c["model"]["dim_s"]
        dim_d = c["model"]["dim_d"]
        dim_a = c["model"]["dim_a"]
        num_classes = c["data"]["num_classes"]
        num_domains = c["data"]["num_domains"]
        meta_cfg = c["data"]["meta"]
        dyn_cfg = c["model"]["dynamics_encoder"]
        acq_cfg = c["model"]["acquisition_encoder"]

        self.semantic_encoder = SemanticEncoder(
            in_dim=D, out_dim=dim_s, hidden_dim=dim_s * 2,
            num_heads=enc_cfg["transformer_heads"],
            dropout=enc_cfg["dropout"],
        )
        self.dynamics_encoder = DynamicsEncoder(
            in_dim=D,
            in_channels=enc_cfg["in_channels"],
            out_dim=dim_d,
            hidden_dim=dim_d * 2,
            use_autocorr=dyn_cfg["use_autocorr"],
            use_spectral=dyn_cfg["use_spectral"],
            spectral_bands=dyn_cfg["spectral_bands"],
        )
        self.acquisition_encoder = AcquisitionEncoder(
            in_dim=D,
            out_dim=dim_a,
            hidden_dim=dim_a * 2,
            d_meta=meta_cfg["d_meta"],
            num_datasets=c["data"]["num_domains"] + 4,  
            num_sr_bins=4,
            num_positions=meta_cfg["num_positions"],
            num_sensor_types=meta_cfg["num_sensor_types"],
        )

        cls_cfg = c["model"]["cls_head"]
        self.cls_head = ClassificationHead(
            dim_s, num_classes,
            hidden_dim=cls_cfg["hidden_dim"],
            dropout=cls_cfg["dropout"],
        )
        self.aux_cls_head = AuxClassificationHead(
            dim_s, dim_d, num_classes,
            hidden_dim=cls_cfg["hidden_dim"],
            dropout=cls_cfg["dropout"],
        )
        proto_cfg = c["model"]["proto_head"]
        self.proto_head = ProtoHead(dim_s, num_classes, temperature=proto_cfg["temperature"])

        self.domain_head_s = DomainHeadOnSemantic(
            dim_s, num_domains,
            max_alpha=c["model"]["grl_lambda_max"],
        )
        self.domain_head_s.grl.set_schedule_steps(c["model"]["grl_schedule_steps"])
        self.domain_head_a = DomainHeadOnAcquisition(dim_a, num_domains)

        self.domain_head_d = DomainHeadOnDynamics(
            dim_d, num_domains,
            max_alpha=c["model"].get("grl_lambda_max_d", 1.5),
        )
        self.domain_head_d.grl.set_schedule_steps(
            c["model"].get("grl_schedule_steps_d", 2000)
        )
        dec_cfg = c["model"]["decoder"]
        self.decoder = FactorDecoder(
            dim_s=dim_s, dim_d=dim_d, dim_a=dim_a,
            hidden_dim=dec_cfg["hidden_dim"],
            out_dim=dec_cfg["out_dim"],
            num_layers=dec_cfg["num_layers"],
        )
        self.swap_reconstructor = SwapReconstructor(self.decoder)
        num_subjects = c["data"].get("num_subjects", 64)
        self.subject_head_d = nn.Sequential(
            nn.Linear(dim_d, dim_d),
            nn.GELU(),
            nn.Linear(dim_d, num_subjects),
        )

        # ── TTA Adapter ───────────────────────────────────────────────────────
        tta_adapter_dim = c.get("model", {}).get("tta_bottleneck_dim", 32)
        self.tta_adapter = AcquisitionAdapter(dim_a, bottleneck_dim=tta_adapter_dim)
        self.tta_adapter.disable()  # 训练时关闭


        self.tta_logit_bias = nn.Linear(dim_a, num_classes, bias=False)
        nn.init.zeros_(self.tta_logit_bias.weight)
        self.tta_logit_bias.weight.requires_grad_(False)  # frozen during training
        self.dim_s = dim_s
        self.dim_d = dim_d
        self.dim_a = dim_a
        self.D = D

    def encode(
        self,
        x_time: torch.Tensor,
        meta: torch.Tensor,
        x_fft: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        enc_out = self.shared_encoder(x_time)
        h_shared = enc_out["h_shared"]  # [B, T, D]
        h_pool = enc_out["h_pool"]      # [B, D]

        z_s = self.semantic_encoder(h_shared)
        z_d = self.dynamics_encoder(h_shared, x_time, x_fft=x_fft)
        z_a = self.acquisition_encoder(h_pool, meta)

        z_a = self.tta_adapter(z_a)

        return {
            "h_shared": h_shared,
            "h_pool": h_pool,
            "z_s": z_s,
            "z_d": z_d,
            "z_a": z_a,
        }

    def forward(
        self,
        x_time: torch.Tensor,
        meta: torch.Tensor,
        x_fft: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        do_swap: bool = False,
    ) -> Dict[str, torch.Tensor]:
 
        enc = self.encode(x_time, meta, x_fft=x_fft)
        z_s, z_d, z_a = enc["z_s"], enc["z_d"], enc["z_a"]
        h_pool = enc["h_pool"]


        logits_main = self.cls_head(z_s)
        if self.tta_adapter._enabled:
            logits_main = logits_main + self.tta_logit_bias(z_a)
        logits_aux = self.aux_cls_head(z_s, z_d)

        proto_out = self.proto_head(z_s)

   
        dom_pred_s = self.domain_head_s(z_s)   # adversarial
        dom_pred_a = self.domain_head_a(z_a)    # direct prediction
        dom_pred_d = self.domain_head_d(z_d)    # adversarial on dynamics

        subj_pred_d = self.subject_head_d(z_d)

        x_rec = self.decoder(z_s, z_d, z_a)    # [B, out_dim]

        out = {
            **enc,
            "logits_main": logits_main,
            "logits_aux": logits_aux,
            "proto_sim": proto_out["sim"],
            "proto_logits": proto_out["logits"],
            "dom_pred_s": dom_pred_s,
            "dom_pred_a": dom_pred_a,
            "dom_pred_d": dom_pred_d,
            "subj_pred_d": subj_pred_d,
            "x_rec": x_rec,
            "rec_target": h_pool.detach(),   
        }

        if do_swap and labels is not None:
            same_idx = build_same_class_pairs(labels)
            swap_out = self.swap_reconstructor(z_s, z_d, z_a, same_idx, swap_type="both")
            out["swap_a"] = swap_out.get("swap_a")
            out["swap_d"] = swap_out.get("swap_d")
            out["swap_idx"] = same_idx

        return out


    def enable_tta(self):
        for p in self.parameters():
            p.requires_grad_(False)


        self.tta_adapter.enable()

        for p in self.tta_logit_bias.parameters():
            p.requires_grad_(True)

    def disable_tta(self):
        for p in self.parameters():
            p.requires_grad_(True)
        self.tta_adapter.disable()

    def grl_step(self):
        self.domain_head_s.step_grl()
        self.domain_head_d.step_grl()

    def get_tta_params(self) -> list:
        params = list(self.tta_adapter.parameters())
        params.extend(self.tta_logit_bias.parameters())
        return params

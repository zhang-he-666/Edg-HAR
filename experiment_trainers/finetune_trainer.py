"""
Fine-tune on target dataset (10%) — Stage 2-4 with lower LR.

Loads the full pretrain checkpoint, then continues with the complete
FactorizedTrainer (Stage 2-4) on the labeled 10% of the target dataset.
"""

import os
import sys
import copy
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional

_TRIFACTOR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "TriFactor-HAR"))
if _TRIFACTOR not in sys.path:
    sys.path.insert(0, _TRIFACTOR)

from models.full_model import FactorizedHARModel
from trainers.factorized_trainer import FactorizedTrainer
from utils.metrics import MetricsTracker
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)


class CrossHARFactorizedFinetuner:
    """
    Fine-tunes the fully pretrained model on 10% of the target dataset.

    Uses FactorizedTrainer (Stage 2-4) with:
      - all model weights loaded from pretrain checkpoint
      - lower learning rate (cfg['finetune']['lr'])
      - fewer epochs per stage (cfg['finetune']['stage*_epochs'])
    """

    def __init__(
        self,
        model: FactorizedHARModel,
        cfg: dict,
        device: str = "cpu",
        output_dir: str = "outputs/crosshar_exp/finetune",
    ):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Build a patched cfg for fine-tuning:
        # override stage epochs and LR with finetune-specific values
        ft_cfg = cfg.get("finetune", {})
        patched = copy.deepcopy(cfg)
        patched["training"]["stage2_epochs"] = ft_cfg.get("stage2_epochs", 20)
        patched["training"]["stage3_epochs"] = ft_cfg.get("stage3_epochs", 15)
        patched["training"]["stage4_epochs"] = ft_cfg.get("stage4_epochs", 15)
        if "lr" in ft_cfg:
            patched["training"]["optimizer"]["lr"] = ft_cfg["lr"]

        self._trainer = FactorizedTrainer(
            model=model,
            cfg=patched,
            device=device,
            output_dir=output_dir,
        )
        self.model = model

    def load_pretrain_ckpt(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint not found: {ckpt_path}  — using current weights.")
            return
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [INFO] Missing keys (expected if proto_ref not in ckpt): {missing[:3]}")
        # Restore proto_ref if saved
        if "proto_ref" in ckpt and ckpt["proto_ref"] is not None:
            self._trainer._proto_ref = ckpt["proto_ref"].to(self.device)
        print(f"  Loaded pretrain checkpoint: {ckpt_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> str:
        ft_cfg = self.cfg.get("finetune", {})
        total = (ft_cfg.get("stage2_epochs", 20)
                 + ft_cfg.get("stage3_epochs", 15)
                 + ft_cfg.get("stage4_epochs", 15))
        print(f"\n=== Fine-tune Stage 2-4 ({total} epochs on 10% target) ===")

        self._trainer.train(
            train_loader, val_loader,
            pretrain_ckpt=None,   # weights already loaded via load_pretrain_ckpt
            start_stage=2,
        )
        best_path = os.path.join(self.output_dir, "best_model.pth")
        return best_path

    def load_best(self):
        best_path = os.path.join(self.output_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best finetune checkpoint: {best_path}")
        else:
            print(f"  [WARN] best_model.pth not found, using final weights.")

    @torch.no_grad()
    def test(self, loader: DataLoader) -> Dict:
        self.model.eval()
        self.model.to(self.device)
        y_true, y_pred = [], []
        for batch in loader:
            x_time = batch["x_time"].to(self.device)
            meta   = batch["meta"].to(self.device)
            labels = batch["y"].to(self.device)
            x_fft  = batch.get("x_fft")
            if x_fft is not None:
                x_fft = x_fft.to(self.device)

            out   = self.model.forward(x_time, meta, x_fft=x_fft)
            preds = out["logits_main"].argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        return {
            "accuracy":    accuracy_score(y_true, y_pred),
            "f1_macro":    f1_score(y_true, y_pred, average="macro",  zero_division=0),
            "f1_micro":    f1_score(y_true, y_pred, average="micro",  zero_division=0),
            "precision":   precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall":      recall_score(y_true, y_pred, average="macro", zero_division=0),
            "conf_matrix": confusion_matrix(y_true, y_pred),
        }

"""
Full pretraining on source datasets — Stage 1-4 complete FactorizedTrainer.

Runs the entire Edge-HAR training pipeline (Stage 1 supervised cls →
Stage 2 adversarial/decorr → Stage 3 prototype/topology → Stage 4 recon/swap)
on 80% of the source datasets.
"""

import os
import sys
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional

_TRIFACTOR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "TriFactor-HAR"))
if _TRIFACTOR not in sys.path:
    sys.path.insert(0, _TRIFACTOR)

from models.full_model import FactorizedHARModel
from trainers.pretrain_trainer import PretrainTrainer
from trainers.factorized_trainer import FactorizedTrainer
from utils.metrics import MetricsTracker


class CrossHARFullPretrainer:
    """
    Runs the complete Edge-HAR training pipeline on source data:
      Stage 1  — shared_encoder + semantic_encoder + cls_head (PretrainTrainer)
      Stage 2-4 — full factorized losses (FactorizedTrainer)

    Saves best_pretrain.pth after Stage 1, best_model.pth after Stage 2-4.
    """

    def __init__(
        self,
        model: FactorizedHARModel,
        cfg: dict,
        device: str = "cpu",
        output_dir: str = "outputs/crosshar_exp/pretrain",
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._pre_trainer = PretrainTrainer(
            model=model,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
        )
        self._fact_trainer = FactorizedTrainer(
            model=model,
            cfg=cfg,
            device=device,
            output_dir=output_dir,
        )

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> str:
        """
        Returns path to the best checkpoint produced after the full pipeline.
        """
        # ── Stage 1 ───────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("PRETRAIN  Stage 1: shared encoder + cls head")
        print("="*60)
        self._pre_trainer.train(train_loader, val_loader)
        stage1_ckpt = os.path.join(self.output_dir, "best_pretrain.pth")

        # ── Stage 2-4 ─────────────────────────────────────────────────────────
        print("\n" + "="*60)
        print("PRETRAIN  Stage 2-4: full factorized pipeline")
        print("="*60)
        self._fact_trainer.train(
            train_loader, val_loader,
            pretrain_ckpt=stage1_ckpt,
            start_stage=2,
        )

        best_path = os.path.join(self.output_dir, "best_model.pth")
        print(f"\nPretraining complete. Best checkpoint: {best_path}")
        return best_path

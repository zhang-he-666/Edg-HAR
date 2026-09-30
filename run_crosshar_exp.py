"""
Main entry point for the CrossHAR-protocol experiment on Edge-HAR.

Usage:
    python run_crosshar_exp.py [--target_id 0|1|2|3]
                               [--config configs/crosshar_exp.yaml]
                               [--device cuda] [--seed 42]
                               [--skip_pretrain]
                               [--finetune_rate 0.02 0.05 0.08 0.10 0.15]

Protocol:
  1. Pretrain   — full Stage 1-4 on 80% of all source datasets
  2. Fine-tune  — Stage 2-4 on X% of target dataset (lower LR, fewer epochs)
  3. Test       — report acc / F1 on remaining (1-X)% of target dataset

Run all 4 leave-one-out experiments with multiple finetune rates:
    python run_crosshar_exp.py --skip_pretrain --finetune_rate 0.02 0.05 0.08 0.10 0.15
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

_THIS      = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
if _TRIFACTOR not in sys.path:
    sys.path.insert(0, _TRIFACTOR)
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

from models.full_model import FactorizedHARModel
from data_loaders.crosshar_datasets import build_crosshar_splits, CROSSHAR_DATASETS
from experiment_trainers.pretrain_trainer import CrossHARFullPretrainer
from experiment_trainers.finetune_trainer import CrossHARFactorizedFinetuner


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def run_one_target(args, cfg: dict, target_id: int, finetune_rate: float):
    tgt_name = CROSSHAR_DATASETS[target_id]["name"]
    rate_tag  = f"ft{int(finetune_rate * 100):02d}pct"
    print("\n" + "=" * 70)
    print(f"  TARGET: {tgt_name}  (id={target_id})  finetune_rate={finetune_rate:.0%}")
    print("=" * 70)

    d_cfg = cfg["data"]
    seed  = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    set_seed(seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    pretrain_ds, finetune_ds, test_ds = build_crosshar_splits(
        data_root=d_cfg["data_root"],
        target_id=target_id,
        pretrain_source_rate=d_cfg["pretrain_source_rate"],
        finetune_target_rate=finetune_rate,
        seed=seed,
        instance_norm=d_cfg.get("instance_norm", True),
    )
    print(f"  pretrain={len(pretrain_ds)} | finetune={len(finetune_ds)} | test={len(test_ds)}")

    if len(finetune_ds) == 0 and finetune_rate > 0:
        print(f"  [WARN] finetune dataset is empty at rate={finetune_rate:.0%}, skipping target {tgt_name}")
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0, "precision": 0.0, "recall": 0.0, "conf_matrix": []}

    bs_tr  = d_cfg["train_batch_size"]
    bs_ft  = d_cfg.get("finetune_batch_size", bs_tr)
    bs_val = d_cfg["val_batch_size"]
    nw     = d_cfg["num_workers"]

    pretrain_loader = make_loader(pretrain_ds, bs_tr,  shuffle=True,  num_workers=nw, drop_last=True)
    # For finetune, never drop last — at low rates (2%) the dataset may be
    # smaller than batch_size, yielding 0 batches with drop_last=True.
    ft_bs = min(bs_ft, max(1, len(finetune_ds))) if finetune_rate > 0 else bs_ft
    finetune_loader = make_loader(finetune_ds, ft_bs,  shuffle=True,  num_workers=nw, drop_last=False) if finetune_rate > 0 else None
    test_loader     = make_loader(test_ds,     bs_val, shuffle=False, num_workers=nw)

    # Use a 20% slice of pretrain data as validation during pretraining
    n_val = max(1, len(pretrain_ds) // 5)
    val_ds = torch.utils.data.Subset(pretrain_ds, list(range(len(pretrain_ds) - n_val, len(pretrain_ds))))
    val_loader = make_loader(val_ds, bs_val, shuffle=False, num_workers=nw)

    # Pretrain ckpt lives at the target dir root (shared across all rates)
    base_dir    = os.path.join(cfg["experiment"]["output_dir"], f"target_{tgt_name}")
    pretrain_dir = os.path.join(base_dir, "pretrain")
    finetune_dir = os.path.join(base_dir, f"finetune_{rate_tag}")
    os.makedirs(base_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FactorizedHARModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params / 1e6:.2f}M")

    # ── Stage 1-4 Pretrain on source ─────────────────────────────────────────
    pretrain_ckpt = os.path.join(pretrain_dir, "best_model.pth")
    # Allow overriding with an explicit checkpoint path
    if getattr(args, "pretrain_ckpt", None):
        pretrain_ckpt = args.pretrain_ckpt

    if not args.skip_pretrain:
        pre_trainer = CrossHARFullPretrainer(
            model=model,
            cfg=cfg,
            device=args.device,
            output_dir=pretrain_dir,
        )
        pretrain_ckpt = pre_trainer.train(pretrain_loader, val_loader)
    else:
        print(f"\n  [Skip pretrain] Will load: {pretrain_ckpt}")
        if os.path.exists(pretrain_ckpt):
            ckpt = torch.load(pretrain_ckpt, map_location=args.device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            print("  Checkpoint loaded.")
        else:
            print("  [WARN] Checkpoint not found — using random init.")

    # ── Stage 2-4 Fine-tune on X% target (skip entirely when rate=0) ──────────
    if finetune_rate == 0.0:
        # Zero-shot: load pretrain ckpt and test directly, no finetuning
        print("\n  [Zero-shot] rate=0%, skipping fine-tune, testing pretrained model directly.")
        if os.path.exists(pretrain_ckpt):
            ckpt = torch.load(pretrain_ckpt, map_location=args.device)
            model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    else:
        ft_trainer = CrossHARFactorizedFinetuner(
            model=model,
            cfg=cfg,
            device=args.device,
            output_dir=finetune_dir,
        )
        ft_trainer.load_pretrain_ckpt(pretrain_ckpt)
        ft_trainer.train(finetune_loader, val_loader=finetune_loader)
        ft_trainer.load_best()

    # ── Test ─────────────────────────────────────────────────────────────────
    # Build a minimal tester that just runs forward pass (works for both 0% and >0%)
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

    model.eval()
    model.to(args.device)
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_time = batch["x_time"].to(args.device)
            meta   = batch["meta"].to(args.device)
            x_fft  = batch.get("x_fft")
            if x_fft is not None:
                x_fft = x_fft.to(args.device)
            out   = model.forward(x_time, meta, x_fft=x_fft)
            preds = out["logits_main"].argmax(dim=1).cpu()
            y_true.extend(batch["y"].tolist())
            y_pred.extend(preds.tolist())
    results = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",  zero_division=0),
        "f1_micro":    f1_score(y_true, y_pred, average="micro",  zero_division=0),
        "precision":   precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":      recall_score(y_true, y_pred, average="macro", zero_division=0),
        "conf_matrix": confusion_matrix(y_true, y_pred),
    }

    print(f"\n{'='*50}")
    print(f"  RESULTS  target={tgt_name}  rate={finetune_rate:.0%}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  F1-macro : {results['f1_macro']:.4f}")
    print(f"  F1-micro : {results['f1_micro']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall   : {results['recall']:.4f}")
    print(f"  Confusion matrix:\n{results['conf_matrix']}")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default=os.path.join(_THIS, "configs", "crosshar_exp.yaml"))
    parser.add_argument("--target_id",     type=int, default=None,
                        help="0=HHAR 1=Motion 2=UCI_HAR 3=Shoaib; omit to run all")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretraining and load existing checkpoint")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Override pretrain checkpoint path (used with --skip_pretrain)")
    parser.add_argument("--finetune_rate", type=float, nargs="+",
                        default=[0.1],
                        help="One or more finetune ratios to sweep, e.g. 0.02 0.05 0.08 0.10 0.15")
    args = parser.parse_args()

    cfg     = load_cfg(args.config)
    targets = [args.target_id] if args.target_id is not None else list(CROSSHAR_DATASETS.keys())
    rates   = args.finetune_rate

    # all_results[rate][tid] = {accuracy, f1_macro, ...}
    all_results: dict = {r: {} for r in rates}

    for tid in targets:
        tgt_name = CROSSHAR_DATASETS[tid]["name"]
        skip_pre = args.skip_pretrain  # can override per-rate below

        for i, rate in enumerate(rates):
            # After the first rate run pretrain is already done for this target
            if i > 0:
                skip_pre = True

            # Temporarily override skip_pretrain so run_one_target reads the right value
            args_copy = argparse.Namespace(**vars(args))
            args_copy.skip_pretrain = skip_pre

            all_results[rate][tid] = run_one_target(args_copy, cfg, tid, rate)

    # ── Cross-rate summary table ──────────────────────────────────────────────
    if len(targets) > 1 or len(rates) > 1:
        rate_headers = "  ".join(f"{r:.0%}".rjust(7) for r in rates)
        print("\n" + "=" * (18 + 9 * len(rates)))
        print(f"  F1-macro  |  {rate_headers}")
        print("-" * (18 + 9 * len(rates)))
        avg_per_rate = {r: [] for r in rates}
        for tid in sorted(targets):
            name = CROSSHAR_DATASETS[tid]["name"]
            row = f"  {name:<12}|"
            for r in rates:
                val = all_results[r].get(tid, {}).get("f1_macro", float("nan"))
                row += f"  {val:>7.4f}"
                avg_per_rate[r].append(val)
            print(row)
        print("-" * (18 + 9 * len(rates)))
        avg_row = f"  {'Average':<12}|"
        for r in rates:
            vals = [v for v in avg_per_rate[r] if not np.isnan(v)]
            avg_row += f"  {np.mean(vals) if vals else float('nan'):>7.4f}"
        print(avg_row)
        print("=" * (18 + 9 * len(rates)))


if __name__ == "__main__":
    main()

"""
PAMAP2 Novel-Class Experiment.

Protocol:
  1. Pretrain  — on HHAR / Motion / UCI_HAR / Shoaib / PAMAP2-seen (4 locomotion classes).
                 PAMAP2 "seen" classes are the same walk/upstairs/downstairs/sit+stand
                 in the existing _new_PAMAP2_windows_w120_h60_c4.npz cache.
  2. Fine-tune — 2% of PAMAP2 novel-class windows (7 never-seen activities:
                 lying, running, cycling, nordic_walking,
                 vacuum_cleaning, ironing, rope_jumping).
                 cls_head is replaced with a fresh 7-way head.
  3. Test      — remaining 98% of novel-class windows.
                 Report overall Acc / F1-macro and per-label Acc + F1.

Usage:
    cd /root/rivermind-data/new
    # Full run (pretrain + finetune + test):
    python run_pamap2_novel_exp.py

    # Skip pretrain if checkpoint already exists:
    python run_pamap2_novel_exp.py --skip_pretrain

    # Sweep multiple fine-tune rates:
    python run_pamap2_novel_exp.py --skip_pretrain --finetune_rate 0.02 0.05 0.10
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

_THIS      = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
_NEW3      = os.path.normpath(os.path.join(_THIS, "..", "new_3"))

for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)
# new_3 for pamap2_dataset loader (12-class)
if _NEW3 not in sys.path:
    sys.path.insert(0, _NEW3)

from models.full_model import FactorizedHARModel
from models.heads import ClassificationHead, ProtoHead

# Import crosshar_datasets directly from this package (new/) to avoid
# collision with new_3/data_loaders which is also on sys.path.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "crosshar_datasets",
    os.path.join(_THIS, "data_loaders", "crosshar_datasets.py"),
)
_crosshar_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_crosshar_mod)
CROSSHAR_DATASETS  = _crosshar_mod.CROSSHAR_DATASETS
load_single_dataset = _crosshar_mod.load_single_dataset

# 12-class loader lives in new_3
from data_loaders.pamap2_dataset import (
    PAMAP2_CLASS_NAMES, PAMAP2_META, load_pamap2_12class,
)
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)

# ── Novel class definition (indices into PAMAP2_CLASS_NAMES 0..11) ────────────
# PAMAP2_CLASS_NAMES = [lying, sitting, standing, walking, running, cycling,
#                       nordic_walking, stairs_up, stairs_down,
#                       vacuum_cleaning, ironing, rope_jumping]
NOVEL_CLASS_INDICES = [0, 4, 5, 6, 9, 10, 11]   # never seen during pretrain
NOVEL_CLASS_NAMES   = [PAMAP2_CLASS_NAMES[i] for i in NOVEL_CLASS_INDICES]
NUM_NOVEL_CLASSES   = len(NOVEL_CLASS_INDICES)   # 7


# ── Novel-class dataset ────────────────────────────────────────────────────────

class PAMAP2NovelDataset(Dataset):
    """
    Wraps the PAMAP2 12-class windows, keeping only novel classes.
    Uses only acc+gyro (first 6 channels of the 9-channel chest IMU)
    so it matches the pretrained 6-channel backbone — no ChannelProjector needed.
    Labels are remapped to 0..NUM_NOVEL_CLASSES-1.
    """

    ACC_GYRO = slice(0, 6)

    def __init__(
        self,
        signals:     np.ndarray,  # [N, 120, 9]
        labels:      np.ndarray,  # [N] in 0..11
        subject_ids: np.ndarray,
        instance_norm: bool = True,
    ):
        orig_to_new = {c: i for i, c in enumerate(NOVEL_CLASS_INDICES)}
        mask = np.isin(labels, NOVEL_CLASS_INDICES)

        sig = signals[mask][:, :, self.ACC_GYRO].astype(np.float32)  # [M,120,6]
        lab = labels[mask]

        if instance_norm:
            t = torch.from_numpy(sig).permute(0, 2, 1)               # [M,6,120]
            t = nn.InstanceNorm1d(6, affine=False)(t)
            sig = t.permute(0, 2, 1).numpy()

        self.signals     = sig
        self.labels      = np.array([orig_to_new[int(l)] for l in lab], dtype=np.int64)
        self.subject_ids = subject_ids[mask].astype(np.int64)

        from datasets.base_dataset import discretize_sr
        sr_bin = discretize_sr(PAMAP2_META["sampling_rate"])
        self._meta = torch.tensor(
            [PAMAP2_META["dataset_id"], sr_bin,
             PAMAP2_META["position"], PAMAP2_META["sensor_type"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        x_time = torch.from_numpy(self.signals[idx]).float().permute(1, 0)  # [6,120]
        return {
            "x_time":     x_time,
            "y":          torch.tensor(self.labels[idx], dtype=torch.long),
            "meta":       self._meta,
            "domain_id":  torch.tensor(PAMAP2_META["dataset_id"], dtype=torch.long),
            "subject_id": torch.tensor(int(self.subject_ids[idx]), dtype=torch.long),
        }


def build_novel_splits(
    data_root: str,
    finetune_rate: float,
    seed: int,
    instance_norm: bool = True,
):
    """Return (finetune_ds, test_ds) over the 7 novel PAMAP2 classes."""
    signals, labels, subject_ids = load_pamap2_12class(data_root)
    ds = PAMAP2NovelDataset(signals, labels, subject_ids, instance_norm=instance_norm)

    rng   = np.random.RandomState(seed)
    n     = len(ds)
    n_ft  = max(1, int(n * finetune_rate))
    idx   = rng.permutation(n)

    ft_ds   = Subset(ds, idx[:n_ft].tolist())
    test_ds = Subset(ds, idx[n_ft:].tolist())

    print(f"  Novel classes ({NUM_NOVEL_CLASSES}): {NOVEL_CLASS_NAMES}")
    print(f"  Total novel windows: {n}  |  finetune: {n_ft}  |  test: {n - n_ft}")
    return ft_ds, test_ds


def build_pretrain_dataset(data_root: str, pretrain_source_rate: float, seed: int, instance_norm: bool):
    """
    Build pretrain dataset from 5 sources:
      HHAR(0), Motion(1), UCI_HAR(2), Shoaib(3) + PAMAP2-seen(4).
    PAMAP2-seen uses the 4-class _new_PAMAP2_windows_w120_h60_c4.npz cache.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for src_id in sorted(CROSSHAR_DATASETS.keys()):
        ds = load_single_dataset(data_root, src_id, instance_norm=instance_norm)
        n  = len(ds)
        n_pretrain = int(n * pretrain_source_rate)
        idx = rng.permutation(n)
        parts.append(Subset(ds, idx[:n_pretrain].tolist()))
        print(f"  Source {CROSSHAR_DATASETS[src_id]['name']}: {n_pretrain}/{n} for pretrain")
    return ConcatDataset(parts)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def replace_cls_head(model: FactorizedHARModel, num_classes: int, cfg: dict):
    cls_cfg = cfg["model"]["cls_head"]
    dim_s   = cfg["model"]["dim_s"]
    model.cls_head = ClassificationHead(
        dim_s, num_classes,
        hidden_dim=cls_cfg["hidden_dim"],
        dropout=cls_cfg["dropout"],
    )
    proto_cfg = cfg["model"]["proto_head"]
    model.proto_head = ProtoHead(dim_s, num_classes, temperature=proto_cfg["temperature"])
    print(f"  cls_head + proto_head replaced for {num_classes} novel classes.")


# ── Pretrain (reuse CrossHARFullPretrainer from new/) ─────────────────────────

def do_pretrain(model, pretrain_loader, val_loader, cfg, device, output_dir):
    from experiment_trainers.pretrain_trainer import CrossHARFullPretrainer
    trainer = CrossHARFullPretrainer(
        model=model, cfg=cfg, device=device, output_dir=output_dir,
    )
    ckpt = trainer.train(pretrain_loader, val_loader)
    return ckpt


# ── Fine-tune: simple CE loop on novel classes ────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, n_correct, n = 0.0, 0, 0
    for batch in loader:
        x_time = batch["x_time"].to(device)
        meta   = batch["meta"].to(device)
        y      = batch["y"].to(device)
        optimizer.zero_grad()
        out  = model(x_time, meta)
        loss = criterion(out["logits_main"], y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n_correct  += (out["logits_main"].argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / max(n, 1), n_correct / max(n, 1)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        x_time = batch["x_time"].to(device)
        meta   = batch["meta"].to(device)
        out    = model(x_time, meta)
        preds  = out["logits_main"].argmax(1).cpu()
        y_true.extend(batch["y"].tolist())
        y_pred.extend(preds.tolist())
    return np.array(y_true), np.array(y_pred)


def do_finetune(model, ft_loader, cfg, device, output_dir):
    """Fine-tune with all parameters; simple CE, cosine LR."""
    ft_cfg    = cfg.get("finetune", {})
    t_cfg     = cfg["training"]
    epochs    = sum([
        ft_cfg.get("stage2_epochs", t_cfg.get("stage2_epochs", 20)),
        ft_cfg.get("stage3_epochs", t_cfg.get("stage3_epochs", 15)),
        ft_cfg.get("stage4_epochs", t_cfg.get("stage4_epochs", 15)),
    ])
    lr        = float(ft_cfg.get("lr", t_cfg["optimizer"]["lr"]))
    wd        = float(t_cfg["optimizer"]["weight_decay"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    min_lr    = float(t_cfg["scheduler"].get("min_lr", 1e-5))
    warmup_ep = int(t_cfg["scheduler"].get("warmup_epochs", 3))

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / max(warmup_ep, 1)
        progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
        return (min_lr / lr) + (1.0 - min_lr / lr) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(output_dir, exist_ok=True)
    best_f1, best_state, best_epoch = 0.0, None, 0

    print(f"\n  Fine-tune: {epochs} epochs, LR={lr:.2e}, wd={wd:.2e}")
    for ep in range(epochs):
        loss, acc = train_one_epoch(model, ft_loader, optimizer, criterion, device, grad_clip)
        scheduler.step()
        y_true, y_pred = evaluate_loader(model, ft_loader, device)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        print(f"  [FT E{ep+1:3d}] loss={loss:.4f}  train_acc={acc:.4f}  f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1 = f1
            best_epoch = ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {"model_state": best_state, "epoch": ep+1, "f1_macro": f1},
                os.path.join(output_dir, "best_model.pth"),
            )

    print(f"  Fine-tune done. Best F1={best_f1:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return os.path.join(output_dir, "best_model.pth")


# ── Per-class report ──────────────────────────────────────────────────────────

def per_class_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    print("\n" + "=" * 76)
    print(f"  {'Class':<22} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'Support':>7}")
    print("-" * 76)

    per = {}
    for i, name in enumerate(class_names):
        support = int((y_true == i).sum())
        if support == 0:
            continue
        tp     = int(cm[i, i])
        acc_i  = tp / support
        prec_i = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        rec_i  = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        f1_i   = 2 * prec_i * rec_i / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0.0
        print(f"  {name:<22} | {acc_i:>7.4f} | {prec_i:>7.4f} | {rec_i:>7.4f} | {f1_i:>7.4f} | {support:>7}")
        per[name] = {"acc": acc_i, "precision": prec_i, "recall": rec_i, "f1": f1_i, "support": support}

    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1    = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    print("-" * 76)
    print(f"  {'Macro avg':<22} | {'':>7} | {'':>7} | {'':>7} | {macro_f1:>7.4f} | {len(y_true):>7}")
    print(f"  {'Overall acc':<22} | {overall_acc:>7.4f} | {'':>7} | {'':>7} | {micro_f1:>7.4f} | {len(y_true):>7}")
    print("=" * 76)

    per["_overall_acc"] = overall_acc
    per["_macro_f1"]    = macro_f1
    per["_micro_f1"]    = micro_f1
    return per


# ── Main ──────────────────────────────────────────────────────────────────────

def run_one_rate(args, cfg: dict, finetune_rate: float):
    rate_tag = f"ft{int(finetune_rate * 100):02d}pct"
    d_cfg = cfg["data"]
    seed  = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    set_seed(seed)

    print("\n" + "=" * 70)
    print(f"  PAMAP2 NOVEL-CLASS  finetune_rate={finetune_rate:.1%}")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    pretrain_ds = build_pretrain_dataset(
        data_root=d_cfg["data_root"],
        pretrain_source_rate=d_cfg["pretrain_source_rate"],
        seed=seed,
        instance_norm=d_cfg.get("instance_norm", True),
    )
    ft_ds, test_ds = build_novel_splits(
        data_root=d_cfg["data_root"],
        finetune_rate=finetune_rate,
        seed=seed,
        instance_norm=d_cfg.get("instance_norm", True),
    )
    print(f"  pretrain={len(pretrain_ds)} | finetune={len(ft_ds)} | test={len(test_ds)}")

    bs_tr  = d_cfg["train_batch_size"]
    bs_ft  = d_cfg.get("finetune_batch_size", bs_tr)
    bs_val = d_cfg["val_batch_size"]
    nw     = d_cfg["num_workers"]

    pretrain_loader = make_loader(pretrain_ds, bs_tr,  shuffle=True,  num_workers=nw, drop_last=True)
    n_val = max(1, len(pretrain_ds) // 5)
    val_ds = Subset(pretrain_ds, list(range(len(pretrain_ds) - n_val, len(pretrain_ds))))
    val_loader = make_loader(val_ds, bs_val, shuffle=False, num_workers=nw)

    ft_bs = min(bs_ft, max(1, len(ft_ds)))
    ft_loader   = make_loader(ft_ds,   ft_bs,  shuffle=True,  num_workers=nw, drop_last=False)
    test_loader = make_loader(test_ds, bs_val, shuffle=False, num_workers=nw)

    # ── Output dirs ───────────────────────────────────────────────────────────
    base_dir     = os.path.join(cfg["experiment"]["output_dir"], "pamap2_novel")
    pretrain_dir = os.path.join(base_dir, "pretrain")
    finetune_dir = os.path.join(base_dir, f"finetune_{rate_tag}")
    os.makedirs(base_dir, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    # Pretrain with the standard 4-class setup; num_domains now 5 (PAMAP2-seen added)
    pretrain_cfg = {k: v for k, v in cfg.items()}
    pretrain_cfg["data"] = dict(cfg["data"])
    pretrain_cfg["data"]["num_classes"] = 4
    pretrain_cfg["data"]["num_domains"] = 5   # 5 source datasets now

    model = FactorizedHARModel(pretrain_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params / 1e6:.2f}M")

    # ── Pretrain ──────────────────────────────────────────────────────────────
    pretrain_ckpt = os.path.join(pretrain_dir, "best_model.pth")
    if getattr(args, "pretrain_ckpt", None):
        pretrain_ckpt = args.pretrain_ckpt

    if not args.skip_pretrain:
        pretrain_ckpt = do_pretrain(model, pretrain_loader, val_loader,
                                    pretrain_cfg, args.device, pretrain_dir)
    else:
        print(f"\n  [Skip pretrain] Will load: {pretrain_ckpt}")
        if os.path.exists(pretrain_ckpt):
            ckpt = torch.load(pretrain_ckpt, map_location=args.device)
            model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
            print("  Checkpoint loaded.")
        else:
            print("  [WARN] Checkpoint not found — using random init.")

    # ── Swap cls_head for 7-way novel classification ──────────────────────────
    replace_cls_head(model, NUM_NOVEL_CLASSES, cfg)
    model = model.to(args.device)

    # ── Fine-tune on novel-class data ─────────────────────────────────────────
    do_finetune(model, ft_loader, cfg, args.device, finetune_dir)

    # ── Test ──────────────────────────────────────────────────────────────────
    y_true, y_pred = evaluate_loader(model, test_loader, args.device)

    print(f"\n{'='*50}")
    print(f"  RESULTS  novel classes  rate={finetune_rate:.1%}")
    print(f"{'='*50}")

    per = per_class_report(y_true, y_pred, NOVEL_CLASS_NAMES)

    out_path = os.path.join(finetune_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump({k: (v if not isinstance(v, np.floating) else float(v))
                   for k, v in per.items()},
                  f, indent=2)
    print(f"\n  Results saved → {out_path}")
    return per


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default=os.path.join(_THIS, "configs", "crosshar_exp.yaml"))
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default=None,
                        help="Override pretrain checkpoint path (used with --skip_pretrain)")
    parser.add_argument("--finetune_rate", type=float, nargs="+", default=[0.02],
                        help="Finetune rate(s) e.g. 0.02 0.05 0.10")
    args = parser.parse_args()

    cfg  = load_cfg(args.config)
    all_results = {}

    for i, rate in enumerate(args.finetune_rate):
        args_i = argparse.Namespace(**vars(args))
        # After first rate, pretrain checkpoint already exists
        if i > 0:
            args_i.skip_pretrain = True
        per = run_one_rate(args_i, cfg, rate)
        all_results[f"{rate:.1%}"] = per

    # ── Summary table ─────────────────────────────────────────────────────────
    if len(args.finetune_rate) > 1:
        rates = args.finetune_rate
        print("\n" + "=" * (22 + 10 * len(rates)))
        header = "  " + "  ".join(f"{r:.1%}".rjust(8) for r in rates)
        print(f"  {'Class':<22}" + "  ".join(f"{r:.1%}".rjust(8) for r in rates))
        print("-" * (22 + 10 * len(rates)))
        for name in NOVEL_CLASS_NAMES:
            row = f"  {name:<22}"
            for r in rates:
                v = all_results.get(f"{r:.1%}", {}).get(name, {})
                row += f"  {v.get('f1', float('nan')):>8.4f}" if isinstance(v, dict) else "       nan"
            print(row)
        print("-" * (22 + 10 * len(rates)))
        macro_row = f"  {'Macro F1':<22}"
        for r in rates:
            macro_row += f"  {all_results.get(f'{r:.1%}', {}).get('_macro_f1', float('nan')):>8.4f}"
        print(macro_row)
        acc_row = f"  {'Overall Acc':<22}"
        for r in rates:
            acc_row += f"  {all_results.get(f'{r:.1%}', {}).get('_overall_acc', float('nan')):>8.4f}"
        print(acc_row)
        print("=" * (22 + 10 * len(rates)))


if __name__ == "__main__":
    main()

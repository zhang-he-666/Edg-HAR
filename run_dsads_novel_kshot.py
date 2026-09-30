"""
DSADS K-shot Novel-Class Adaptation Experiment.

Protocol:
  1. Pretrain  — HHAR / Motion / UCI_HAR / Shoaib / PAMAP2-seen (4-class caches)
                 All share the same 4-class (walk/upstairs/downstairs/sit+stand) label space.
  2. Fine-tune — K shots per class from DSADS novel classes (classes not seen in pretrain).
                 cls_head is replaced with a fresh N_novel-way head.
                 K ∈ {20, 50, 100} by default.
  3. Test      — Remaining DSADS novel windows (per class = 792 - K).
                 Report overall Acc / F1-macro and per-label Acc + F1.

DSADS-19 class split (label indices 0-18):
  SEEN    (locomotion analogues, in pretrain label space):
    0=sitting, 1=standing, 4=ascending_stairs, 5=descending_stairs,
    9=walking_treadmill_4kmh_flat  → map to pretrain classes
  NOVEL   (never seen during pretrain, 14 classes):
    2=lying_back, 3=lying_right, 6=standing_in_elevator, 7=moving_in_elevator,
    8=walking_in_parking_lot, 10=walking_treadmill_4kmh_inclined,
    11=running_on_treadmill_8kmh, 12=exercising_on_stepper,
    13=exercising_on_cross_trainer, 14=cycling_exercise_bike_horizontal,
    15=cycling_exercise_bike_vertical, 16=rowing, 17=jumping, 18=playing_basketball

  Note: walking_treadmill (label 9) is a locomotion seen-analogue;
        walking_parking_lot (label 8) looks like walking but different sensor context
        → we keep label 8 in novel to test a "familiar motion, new context" case.

Usage:
    cd /root/rivermind-data/new

    # Full run with K=20,50,100:
    python run_dsads_novel_kshot.py

    # Skip pretrain (reuse checkpoint from pamap2_novel exp):
    python run_dsads_novel_kshot.py --skip_pretrain \
        --pretrain_ckpt outputs/crosshar_exp/pamap2_novel/pretrain/best_model.pth

    # Custom K:
    python run_dsads_novel_kshot.py --skip_pretrain --k_shots 10 20 50 100
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
_TRIFACTOR1 = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR_1"))

for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.full_model import FactorizedHARModel
from models.heads import ClassificationHead, ProtoHead
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score,
)

# ── DSADS-19 class definitions ────────────────────────────────────────────────
# Cache: Edge-HAR_1/Cross_person/_cp_DSADS_windows_w120_h60_c19.npz
# Labels 0..18 correspond to activities a01..a19 remapped to 0-based.
DSADS_ALL_NAMES = [
    "sitting",                           # 0  ← seen (sit+stand analogue)
    "standing",                          # 1  ← seen (sit+stand analogue)
    "lying_back",                        # 2  NOVEL
    "lying_right",                       # 3  NOVEL
    "ascending_stairs",                  # 4  ← seen (upstairs analogue)
    "descending_stairs",                 # 5  ← seen (downstairs analogue)
    "standing_in_elevator",              # 6  NOVEL
    "moving_in_elevator",                # 7  NOVEL
    "walking_parking_lot",               # 8  NOVEL (walk but novel context)
    "walking_treadmill_4kmh",            # 9  ← seen (walk analogue)
    "walking_treadmill_inclined",        # 10 NOVEL
    "running_treadmill_8kmh",            # 11 NOVEL
    "exercising_stepper",                # 12 NOVEL
    "exercising_cross_trainer",          # 13 NOVEL
    "cycling_bike_horizontal",           # 14 NOVEL
    "cycling_bike_vertical",             # 15 NOVEL
    "rowing",                            # 16 NOVEL
    "jumping",                           # 17 NOVEL
    "playing_basketball",                # 18 NOVEL
]

# Seen classes (analogues to pretrain 4-class label space) — excluded from novel task
DSADS_SEEN_INDICES  = [0, 1, 4, 5, 9]
# Novel classes used in the K-shot experiment
DSADS_NOVEL_INDICES = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
NOVEL_CLASS_NAMES   = [DSADS_ALL_NAMES[i] for i in DSADS_NOVEL_INDICES]
NUM_NOVEL_CLASSES   = len(DSADS_NOVEL_INDICES)  # 14

DSADS_CACHE = os.path.join(
    _TRIFACTOR1, "Cross_person", "_cp_DSADS_windows_w120_h60_c19.npz"
)

# DSADS meta (from cross_person_exp: dataset_id=0, sr=20, position=chest, acc+gyro)
from datasets.base_dataset import POSITION_MAP, SENSOR_TYPE_MAP, discretize_sr
_DSADS_META = torch.tensor(
    [0, discretize_sr(20), POSITION_MAP["chest"], SENSOR_TYPE_MAP["acc+gyro"]],
    dtype=torch.long,
)


# ── Dataset class ─────────────────────────────────────────────────────────────

class DSADSNovelDataset(Dataset):
    """DSADS novel classes only; labels remapped to 0..NUM_NOVEL_CLASSES-1."""

    def __init__(self, signals, labels, subject_ids, instance_norm=True):
        orig_to_new = {c: i for i, c in enumerate(DSADS_NOVEL_INDICES)}
        mask = np.isin(labels, DSADS_NOVEL_INDICES)

        sig = signals[mask].astype(np.float32)   # [M, 120, 6]
        lab = labels[mask]

        if instance_norm:
            t = torch.from_numpy(sig).permute(0, 2, 1)
            t = nn.InstanceNorm1d(6, affine=False)(t)
            sig = t.permute(0, 2, 1).numpy()

        self.signals     = sig
        self.labels      = np.array([orig_to_new[int(l)] for l in lab], dtype=np.int64)
        self.subject_ids = subject_ids[mask].astype(np.int64)
        # class_to_indices for K-shot sampling
        self.class_to_indices = {
            i: np.where(self.labels == i)[0]
            for i in range(NUM_NOVEL_CLASSES)
        }

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x_time = torch.from_numpy(self.signals[idx]).float().permute(1, 0)  # [6,120]
        return {
            "x_time":     x_time,
            "y":          torch.tensor(self.labels[idx], dtype=torch.long),
            "meta":       _DSADS_META,
            "domain_id":  torch.tensor(0, dtype=torch.long),
            "subject_id": torch.tensor(int(self.subject_ids[idx]), dtype=torch.long),
        }


def load_dsads_novel(instance_norm=True):
    d = np.load(DSADS_CACHE)
    ds = DSADSNovelDataset(
        d["signals"], d["labels"], d["subject_ids"],
        instance_norm=instance_norm,
    )
    print(f"  [DSADS novel] {len(ds)} windows, {NUM_NOVEL_CLASSES} classes")
    for i, name in enumerate(NOVEL_CLASS_NAMES):
        print(f"    {name:<35}: {len(ds.class_to_indices[i])} windows")
    return ds


def build_kshot_splits(ds: DSADSNovelDataset, k: int, seed: int):
    """
    K-shot: exactly K windows per novel class for fine-tune,
    all remaining windows for test.
    Returns (ft_indices, test_indices).
    """
    rng = np.random.RandomState(seed)
    ft_idx, test_idx = [], []
    for cls_i, pool in ds.class_to_indices.items():
        if len(pool) < k:
            raise ValueError(
                f"Class {NOVEL_CLASS_NAMES[cls_i]} has only {len(pool)} windows "
                f"but K={k} requested."
            )
        chosen = rng.choice(pool, size=k, replace=False)
        rest   = np.setdiff1d(pool, chosen)
        ft_idx.extend(chosen.tolist())
        test_idx.extend(rest.tolist())
    return ft_idx, test_idx


# ── Pretrain data ─────────────────────────────────────────────────────────────

def _load_crosshar_datasets():
    """Load CROSSHAR_DATASETS and load_single_dataset from new/data_loaders."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "crosshar_datasets",
        os.path.join(_THIS, "data_loaders", "crosshar_datasets.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.CROSSHAR_DATASETS, m.load_single_dataset


def build_pretrain_dataset(data_root, pretrain_source_rate, seed, instance_norm):
    CROSSHAR_DATASETS, load_single_dataset = _load_crosshar_datasets()
    rng = np.random.RandomState(seed)
    parts = []
    for src_id in sorted(CROSSHAR_DATASETS.keys()):
        ds = load_single_dataset(data_root, src_id, instance_norm=instance_norm)
        n  = len(ds)
        n_pre = int(n * pretrain_source_rate)
        idx = rng.permutation(n)
        parts.append(Subset(ds, idx[:n_pre].tolist()))
        print(f"  Source {CROSSHAR_DATASETS[src_id]['name']}: {n_pre}/{n} for pretrain")
    return ConcatDataset(parts)


# ── Model helpers ─────────────────────────────────────────────────────────────

def replace_cls_head(model, num_classes, cfg):
    cls_cfg   = cfg["model"]["cls_head"]
    proto_cfg = cfg["model"]["proto_head"]
    dim_s     = cfg["model"]["dim_s"]
    model.cls_head = ClassificationHead(
        dim_s, num_classes,
        hidden_dim=cls_cfg["hidden_dim"],
        dropout=cls_cfg["dropout"],
    )
    model.proto_head = ProtoHead(dim_s, num_classes, temperature=proto_cfg["temperature"])
    print(f"  cls_head + proto_head replaced → {num_classes} novel classes.")


def load_pretrain_ckpt(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        print(f"  [WARN] No checkpoint at {ckpt_path} — random init.")
        return
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    cur   = model.state_dict()
    filt  = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
    model.load_state_dict(filt, strict=False)
    print(f"  Loaded pretrain ckpt: {ckpt_path}  ({len(filt)}/{len(cur)} layers matched)")


# ── Training helpers ──────────────────────────────────────────────────────────

def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
    )


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, n_corr, n = 0.0, 0, 0
    for batch in loader:
        x = batch["x_time"].to(device)
        m = batch["meta"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        out  = model(x, m)
        loss = criterion(out["logits_main"], y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n_corr     += (out["logits_main"].argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / max(n, 1), n_corr / max(n, 1)


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        out  = model(batch["x_time"].to(device), batch["meta"].to(device))
        preds = out["logits_main"].argmax(1).cpu()
        yt.extend(batch["y"].tolist())
        yp.extend(preds.tolist())
    return np.array(yt), np.array(yp)


def finetune(model, ft_loader, cfg, device, output_dir, num_novel):
    ft_cfg    = cfg.get("finetune", {})
    t_cfg     = cfg["training"]
    epochs    = (ft_cfg.get("stage2_epochs", t_cfg.get("stage2_epochs", 20))
               + ft_cfg.get("stage3_epochs", t_cfg.get("stage3_epochs", 15))
               + ft_cfg.get("stage4_epochs", t_cfg.get("stage4_epochs", 15)))
    lr        = float(ft_cfg.get("lr", t_cfg["optimizer"]["lr"]))
    wd        = float(t_cfg["optimizer"]["weight_decay"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    min_lr    = float(t_cfg["scheduler"].get("min_lr", 1e-5))
    warmup_ep = int(t_cfg["scheduler"].get("warmup_epochs", 3))

    # Class-weighted CE to handle any remaining class imbalance in the K-shot batch
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / max(warmup_ep, 1)
        progress = (ep - warmup_ep) / max(1, epochs - warmup_ep)
        return (min_lr / lr) + (1.0 - min_lr / lr) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(output_dir, exist_ok=True)

    best_f1, best_state, best_ep = 0.0, None, 0
    print(f"  Fine-tune: {epochs} epochs, LR={lr:.2e}")
    for ep in range(epochs):
        loss, acc = train_epoch(model, ft_loader, optimizer, criterion, device, grad_clip)
        scheduler.step()
        yt, yp = eval_loader(model, ft_loader, device)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"  [FT E{ep+1:3d}] loss={loss:.4f}  train_acc={acc:.4f}  f1={f1:.4f}")
        if f1 >= best_f1:
            best_f1, best_ep = f1, ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_state": best_state, "epoch": ep+1, "f1_macro": f1},
                       os.path.join(output_dir, "best_model.pth"))

    print(f"  Best train-set F1={best_f1:.4f} @ epoch {best_ep}")
    if best_state is not None:
        model.load_state_dict(best_state)


# ── Per-class report ──────────────────────────────────────────────────────────

def per_class_report(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print("\n" + "=" * 82)
    print(f"  {'Class':<35} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | "
          f"{'F1':>7} | {'Support':>7}")
    print("-" * 82)
    per = {}
    for i, name in enumerate(class_names):
        support = int((y_true == i).sum())
        if support == 0:
            continue
        tp    = int(cm[i, i])
        acc_i = tp / support
        p_i   = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        r_i   = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        f1_i  = 2 * p_i * r_i / (p_i + r_i) if (p_i + r_i) > 0 else 0.0
        print(f"  {name:<35} | {acc_i:>7.4f} | {p_i:>7.4f} | {r_i:>7.4f} | "
              f"{f1_i:>7.4f} | {support:>7}")
        per[name] = {"acc": acc_i, "precision": float(p_i),
                     "recall": float(r_i), "f1": float(f1_i), "support": support}

    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1    = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    print("-" * 82)
    print(f"  {'Macro avg':<35} | {'':>7} | {'':>7} | {'':>7} | "
          f"{macro_f1:>7.4f} | {len(y_true):>7}")
    print(f"  {'Overall acc':<35} | {overall_acc:>7.4f} | {'':>7} | {'':>7} | "
          f"{micro_f1:>7.4f} | {len(y_true):>7}")
    print("=" * 82)
    per["_overall_acc"] = float(overall_acc)
    per["_macro_f1"]    = float(macro_f1)
    per["_micro_f1"]    = float(micro_f1)
    return per


# ── Main ──────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_one_k(args, cfg, k, novel_ds, pretrain_ckpt_path):
    seed = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    set_seed(seed)
    d_cfg = cfg["data"]

    print(f"\n{'='*70}")
    print(f"  K-SHOT = {k}  ({k} per class × {NUM_NOVEL_CLASSES} classes = {k*NUM_NOVEL_CLASSES} finetune windows)")
    print(f"{'='*70}")

    ft_idx, test_idx = build_kshot_splits(novel_ds, k, seed)
    ft_ds   = Subset(novel_ds, ft_idx)
    test_ds = Subset(novel_ds, test_idx)
    print(f"  finetune={len(ft_ds)}  test={len(test_ds)}")

    bs_ft  = min(d_cfg.get("finetune_batch_size", d_cfg["train_batch_size"]),
                 max(1, len(ft_ds)))
    bs_val = d_cfg["val_batch_size"]
    nw     = d_cfg["num_workers"]
    ft_loader   = make_loader(ft_ds,   bs_ft,  shuffle=True,  num_workers=nw)
    test_loader = make_loader(test_ds, bs_val, shuffle=False, num_workers=nw)

    # Build model (same 5-source pretrain config: 4 classes, 5 domains)
    pretrain_cfg = {k_: v for k_, v in cfg.items()}
    pretrain_cfg["data"] = dict(cfg["data"])
    pretrain_cfg["data"]["num_classes"] = 4
    pretrain_cfg["data"]["num_domains"] = 5

    model = FactorizedHARModel(pretrain_cfg)
    load_pretrain_ckpt(model, pretrain_ckpt_path, args.device)
    replace_cls_head(model, NUM_NOVEL_CLASSES, cfg)
    model = model.to(args.device)

    out_dir = os.path.join(
        cfg["experiment"]["output_dir"], "dsads_novel_kshot", f"k{k:03d}"
    )
    finetune(model, ft_loader, cfg, args.device, out_dir, NUM_NOVEL_CLASSES)

    y_true, y_pred = eval_loader(model, test_loader, args.device)

    print(f"\n--- RESULTS  K={k} ---")
    per = per_class_report(y_true, y_pred, NOVEL_CLASS_NAMES)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(per, f, indent=2)
    print(f"  Saved → {out_dir}/results.json")
    return per


def main():
    default_ckpt = os.path.join(
        _THIS, "outputs", "crosshar_exp", "pamap2_novel", "pretrain", "best_model.pth"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default=os.path.join(_THIS, "configs", "crosshar_exp.yaml"))
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--k_shots",       type=int, nargs="+", default=[20, 50, 100])
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    d_cfg = cfg["data"]
    seed  = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    set_seed(seed)

    # ── Pretrain ──────────────────────────────────────────────────────────────
    pretrain_dir  = os.path.join(cfg["experiment"]["output_dir"], "dsads_novel_kshot", "pretrain")
    pretrain_ckpt = os.path.join(pretrain_dir, "best_model.pth")
    if args.pretrain_ckpt:
        pretrain_ckpt = args.pretrain_ckpt

    if not args.skip_pretrain:
        print("\nLoading pretrain sources ...")
        pretrain_ds = build_pretrain_dataset(
            d_cfg["data_root"], d_cfg["pretrain_source_rate"], seed,
            d_cfg.get("instance_norm", True),
        )
        bs_tr  = d_cfg["train_batch_size"]
        bs_val = d_cfg["val_batch_size"]
        nw     = d_cfg["num_workers"]

        pretrain_loader = make_loader(pretrain_ds, bs_tr, shuffle=True,
                                      num_workers=nw, drop_last=True)
        n_val  = max(1, len(pretrain_ds) // 5)
        val_ds = Subset(pretrain_ds,
                        list(range(len(pretrain_ds) - n_val, len(pretrain_ds))))
        val_loader = make_loader(val_ds, bs_val, shuffle=False, num_workers=nw)

        pretrain_cfg = {k_: v for k_, v in cfg.items()}
        pretrain_cfg["data"] = dict(cfg["data"])
        pretrain_cfg["data"]["num_classes"] = 4
        pretrain_cfg["data"]["num_domains"] = 5
        model = FactorizedHARModel(pretrain_cfg)

        from experiment_trainers.pretrain_trainer import CrossHARFullPretrainer
        trainer = CrossHARFullPretrainer(
            model=model, cfg=pretrain_cfg, device=args.device, output_dir=pretrain_dir,
        )
        pretrain_ckpt = trainer.train(pretrain_loader, val_loader)
        print(f"  Pretrain done. Ckpt: {pretrain_ckpt}")
    else:
        print(f"\n[Skip pretrain] Using ckpt: {pretrain_ckpt}")
        if not os.path.exists(pretrain_ckpt):
            print("  [WARN] Checkpoint not found.")

    # ── Load DSADS novel dataset once ─────────────────────────────────────────
    print("\nLoading DSADS novel classes ...")
    novel_ds = load_dsads_novel(instance_norm=d_cfg.get("instance_norm", True))

    # ── K-shot sweep ──────────────────────────────────────────────────────────
    all_results = {}
    for k in args.k_shots:
        per = run_one_k(args, cfg, k, novel_ds, pretrain_ckpt)
        all_results[k] = per

    # ── Summary table across K values ─────────────────────────────────────────
    if len(args.k_shots) > 1:
        ks = args.k_shots
        col_w = 9
        sep = "=" * (38 + col_w * len(ks))
        print("\n" + sep)
        header = f"  {'Class':<35} " + "".join(f"K={k:>5} " for k in ks)
        print(header)
        print("-" * (38 + col_w * len(ks)))
        for name in NOVEL_CLASS_NAMES:
            row = f"  {name:<35} "
            for k in ks:
                v = all_results.get(k, {}).get(name, {})
                row += f"{v.get('f1', float('nan')):>7.4f}  " if isinstance(v, dict) else "    nan  "
            print(row)
        print("-" * (38 + col_w * len(ks)))
        macro_row = f"  {'Macro F1':<35} "
        acc_row   = f"  {'Overall Acc':<35} "
        for k in ks:
            macro_row += f"{all_results.get(k, {}).get('_macro_f1', float('nan')):>7.4f}  "
            acc_row   += f"{all_results.get(k, {}).get('_overall_acc', float('nan')):>7.4f}  "
        print(macro_row)
        print(acc_row)
        print(sep)

        # Save summary
        summary_path = os.path.join(
            cfg["experiment"]["output_dir"], "dsads_novel_kshot", "summary.json"
        )
        with open(summary_path, "w") as f:
            json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
        print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()

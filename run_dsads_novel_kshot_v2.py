"""
DSADS K-shot Novel-Class Experiment v2 — Frozen-backbone strategy.

Key difference from v1: instead of full-parameter fine-tuning,
we freeze the pretrained backbone and only train a fresh linear head.
This is the correct inductive few-shot setup:
  - Backbone stays fixed → exploits pretrained representations
  - Only cls_head learns → no catastrophic forgetting with tiny data

Two strategies compared per K:
  A. frozen   — backbone frozen, only cls_head trained (200 epochs, higher LR)
  B. full_ft  — all params fine-tuned (50 epochs, lower LR) — same as v1
"""

import argparse, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import yaml

_THIS       = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR  = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
_TRIFACTOR1 = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR_1"))
for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.full_model import FactorizedHARModel
from models.heads import ClassificationHead, ProtoHead
from torch.utils.data import Dataset, Subset, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score

from datasets.base_dataset import POSITION_MAP, SENSOR_TYPE_MAP, discretize_sr

DSADS_NOVEL_INDICES = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
NOVEL_CLASS_NAMES = [
    "lying_back", "lying_right", "standing_in_elevator", "moving_in_elevator",
    "walking_parking_lot", "walking_treadmill_inclined", "running_treadmill_8kmh",
    "exercising_stepper", "exercising_cross_trainer",
    "cycling_bike_horizontal", "cycling_bike_vertical",
    "rowing", "jumping", "playing_basketball",
]
NUM_NOVEL = len(DSADS_NOVEL_INDICES)

DSADS_CACHE = os.path.join(_TRIFACTOR1, "Cross_person", "_cp_DSADS_windows_w120_h60_c19.npz")
_DSADS_META = torch.tensor(
    [0, discretize_sr(20), POSITION_MAP["chest"], SENSOR_TYPE_MAP["acc+gyro"]],
    dtype=torch.long,
)


class DSADSNovelDataset(Dataset):
    def __init__(self, signals, labels, subject_ids, instance_norm=True):
        orig_to_new = {c: i for i, c in enumerate(DSADS_NOVEL_INDICES)}
        mask = np.isin(labels, DSADS_NOVEL_INDICES)
        sig  = signals[mask].astype(np.float32)
        lab  = labels[mask]
        if instance_norm:
            t = torch.from_numpy(sig).permute(0, 2, 1)
            t = nn.InstanceNorm1d(6, affine=False)(t)
            sig = t.permute(0, 2, 1).numpy()
        self.signals     = sig
        self.labels      = np.array([orig_to_new[int(l)] for l in lab], dtype=np.int64)
        self.subject_ids = subject_ids[mask].astype(np.int64)
        self.class_to_indices = {i: np.where(self.labels == i)[0] for i in range(NUM_NOVEL)}

    def __len__(self): return len(self.signals)

    def __getitem__(self, idx):
        return {
            "x_time":     torch.from_numpy(self.signals[idx]).float().permute(1, 0),
            "y":          torch.tensor(self.labels[idx], dtype=torch.long),
            "meta":       _DSADS_META,
            "domain_id":  torch.tensor(0, dtype=torch.long),
            "subject_id": torch.tensor(int(self.subject_ids[idx]), dtype=torch.long),
        }


def load_dsads_novel(instance_norm=True):
    d  = np.load(DSADS_CACHE)
    ds = DSADSNovelDataset(d["signals"], d["labels"], d["subject_ids"], instance_norm)
    print(f"  [DSADS novel] {len(ds)} windows, {NUM_NOVEL} classes, 792/class")
    return ds


def build_kshot_splits(ds, k, seed):
    rng = np.random.RandomState(seed)
    ft_idx, test_idx = [], []
    for i, pool in ds.class_to_indices.items():
        chosen = rng.choice(pool, size=k, replace=False)
        ft_idx.extend(chosen.tolist())
        test_idx.extend(np.setdiff1d(pool, chosen).tolist())
    return ft_idx, test_idx


def make_loader(ds, batch_size, shuffle, num_workers=4, drop_last=False):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last,
    )


def build_model(cfg, pretrain_ckpt, device):
    pcfg = {k: v for k, v in cfg.items()}
    pcfg["data"] = dict(cfg["data"])
    pcfg["data"]["num_classes"] = 4
    pcfg["data"]["num_domains"] = 5
    model = FactorizedHARModel(pcfg)
    # Load pretrain weights
    if os.path.exists(pretrain_ckpt):
        ckpt  = torch.load(pretrain_ckpt, map_location=device)
        state = ckpt.get("model_state", ckpt)
        cur   = model.state_dict()
        filt  = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
        print(f"  Loaded ckpt: {pretrain_ckpt}  ({len(filt)}/{len(cur)} layers)")
    else:
        print(f"  [WARN] No ckpt at {pretrain_ckpt}")
    # Replace head
    dim_s = cfg["model"]["dim_s"]
    cls_cfg = cfg["model"]["cls_head"]
    proto_cfg = cfg["model"]["proto_head"]
    model.cls_head  = ClassificationHead(dim_s, NUM_NOVEL,
                                          hidden_dim=cls_cfg["hidden_dim"],
                                          dropout=cls_cfg["dropout"])
    model.proto_head = ProtoHead(dim_s, NUM_NOVEL, temperature=proto_cfg["temperature"])
    return model.to(device)


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("cls_head.")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Frozen backbone — trainable: {n/1e3:.1f}K params (cls_head only)")


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, n_corr, n = 0.0, 0, 0
    for batch in loader:
        x, m, y = batch["x_time"].to(device), batch["meta"].to(device), batch["y"].to(device)
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
        out   = model(batch["x_time"].to(device), batch["meta"].to(device))
        preds = out["logits_main"].argmax(1).cpu()
        yt.extend(batch["y"].tolist()); yp.extend(preds.tolist())
    return np.array(yt), np.array(yp)


def run_finetune(model, ft_loader, device, strategy, cfg, out_dir):
    """
    strategy: "frozen"  → freeze backbone, 200 epochs, LR=1e-3
              "full_ft" → all params, 50 epochs, LR=5e-4
    """
    if strategy == "frozen":
        freeze_backbone(model)
        epochs, lr = 200, 1e-3
    else:
        unfreeze_all(model)
        epochs, lr = 50, 5e-4

    t_cfg  = cfg["training"]
    wd     = float(t_cfg["optimizer"]["weight_decay"])
    clip   = float(t_cfg.get("grad_clip", 1.0))
    min_lr = float(t_cfg["scheduler"].get("min_lr", 1e-5))
    warmup = min(5, epochs // 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd
    )

    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(1, epochs - warmup)
        return (min_lr / lr) + (1 - min_lr / lr) * 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(out_dir, exist_ok=True)

    best_f1, best_state, best_ep = 0.0, None, 0
    log_interval = max(1, epochs // 10)
    print(f"  [{strategy}] {epochs} epochs, LR={lr:.2e}")

    for ep in range(epochs):
        loss, acc = train_epoch(model, ft_loader, optimizer, criterion, device, clip)
        scheduler.step()
        if (ep + 1) % log_interval == 0 or ep == epochs - 1:
            yt, yp = eval_loader(model, ft_loader, device)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            print(f"  [E{ep+1:4d}] loss={loss:.4f}  train_acc={acc:.4f}  train_f1={f1:.4f}")
            if f1 >= best_f1:
                best_f1, best_ep = f1, ep + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model_state": best_state, "epoch": ep+1, "f1_macro": f1},
                           os.path.join(out_dir, "best_model.pth"))

    print(f"  Best train F1={best_f1:.4f} @ epoch {best_ep}")
    if best_state:
        model.load_state_dict(best_state)


def per_class_report(y_true, y_pred, class_names, tag=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print(f"\n{'='*82}")
    if tag: print(f"  {tag}")
    print(f"  {'Class':<35} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'N':>6}")
    print(f"  {'-'*80}")
    per = {}
    for i, name in enumerate(class_names):
        support = int((y_true == i).sum())
        if support == 0: continue
        acc_i = cm[i, i] / support
        p_i   = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        r_i   = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        f1_i  = 2*p_i*r_i/(p_i+r_i) if (p_i+r_i) > 0 else 0.0
        print(f"  {name:<35} | {acc_i:>7.4f} | {p_i:>7.4f} | {r_i:>7.4f} | {f1_i:>7.4f} | {support:>6}")
        per[name] = {"acc": float(acc_i), "precision": float(p_i),
                     "recall": float(r_i), "f1": float(f1_i), "support": support}
    acc   = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"  {'-'*80}")
    print(f"  {'Macro avg':<35} | {'':>7} | {'':>7} | {'':>7} | {macro:>7.4f} | {len(y_true):>6}")
    print(f"  {'Overall acc':<35} | {acc:>7.4f} | {'':>7} | {'':>7} | {micro:>7.4f} | {len(y_true):>6}")
    print("="*82)
    per.update({"_overall_acc": float(acc), "_macro_f1": float(macro), "_micro_f1": float(micro)})
    return per


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    default_ckpt = os.path.join(
        _THIS, "outputs", "crosshar_exp", "pamap2_novel", "pretrain", "best_model.pth"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default=os.path.join(_THIS, "configs", "crosshar_exp.yaml"))
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",          type=int, default=None)
    parser.add_argument("--pretrain_ckpt", type=str, default=default_ckpt)
    parser.add_argument("--k_shots",       type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--strategies",    type=str, nargs="+",
                        default=["frozen", "full_ft"],
                        help="frozen | full_ft")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg["experiment"]["seed"]
    set_seed(seed)

    print("\nLoading DSADS novel classes ...")
    novel_ds = load_dsads_novel(cfg["data"].get("instance_norm", True))

    all_results = {}   # [strategy][k] = per-class dict
    for strategy in args.strategies:
        all_results[strategy] = {}
        for k in args.k_shots:
            set_seed(seed)
            print(f"\n{'='*70}")
            print(f"  Strategy={strategy}  K={k}  "
                  f"({k}×{NUM_NOVEL}={k*NUM_NOVEL} FT windows, "
                  f"{792-k}×{NUM_NOVEL}={(792-k)*NUM_NOVEL} test windows)")
            print("="*70)

            ft_idx, test_idx = build_kshot_splits(novel_ds, k, seed)
            ft_ds   = Subset(novel_ds, ft_idx)
            test_ds = Subset(novel_ds, test_idx)

            bs_ft  = min(128, max(1, len(ft_ds)))
            nw     = cfg["data"]["num_workers"]
            ft_loader   = make_loader(ft_ds,   bs_ft, shuffle=True,  num_workers=nw)
            test_loader = make_loader(test_ds, 256,   shuffle=False, num_workers=nw)

            model   = build_model(cfg, args.pretrain_ckpt, args.device)
            out_dir = os.path.join(
                cfg["experiment"]["output_dir"],
                "dsads_novel_kshot_v2", strategy, f"k{k:03d}"
            )
            run_finetune(model, ft_loader, args.device, strategy, cfg, out_dir)

            y_true, y_pred = eval_loader(model, test_loader, args.device)
            per = per_class_report(y_true, y_pred, NOVEL_CLASS_NAMES,
                                   tag=f"strategy={strategy}  K={k}")
            all_results[strategy][k] = per

            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(per, f, indent=2)
            print(f"  Saved → {out_dir}/results.json")

    # ── Summary table (strategy=frozen, metric=F1 per class) ─────────────────
    for strategy in args.strategies:
        ks = args.k_shots
        print(f"\n{'='*82}")
        print(f"  SUMMARY  strategy={strategy}  (F1 per class)")
        print(f"  {'Class':<35} " + "".join(f"  K={k:>4}" for k in ks))
        print(f"  {'-'*80}")
        for name in NOVEL_CLASS_NAMES:
            row = f"  {name:<35} "
            for k in ks:
                v = all_results[strategy].get(k, {}).get(name, {})
                row += f"  {v.get('f1', float('nan')):>6.4f}" if isinstance(v, dict) else "     nan"
            print(row)
        print(f"  {'-'*80}")
        macro_row = f"  {'Macro F1':<35} "
        acc_row   = f"  {'Overall Acc':<35} "
        for k in ks:
            macro_row += f"  {all_results[strategy].get(k, {}).get('_macro_f1', float('nan')):>6.4f}"
            acc_row   += f"  {all_results[strategy].get(k, {}).get('_overall_acc', float('nan')):>6.4f}"
        print(macro_row); print(acc_row)
        print("="*82)

    summary_path = os.path.join(
        cfg["experiment"]["output_dir"], "dsads_novel_kshot_v2", "summary.json"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({s: {str(k): v for k, v in res.items()}
                   for s, res in all_results.items()}, f, indent=2)
    print(f"\nFull summary → {summary_path}")


if __name__ == "__main__":
    main()

"""
Novel-class adaptation strategy comparison.

Same 5 strategies as run_adapt_strategy_exp.py but on DSADS 14 novel classes
(never seen during pretraining).  K-shot: K ∈ {20, 50, 100}.

Strategies:
  A. linear_probe  — freeze all backbone, train fresh cls_head only
  B. adapter       — freeze backbone, 2-layer residual MLP after semantic_encoder
  C. partial       — freeze shared_encoder, fine-tune semantic_encoder + cls_head
  D. full_ft       — all params, pretrained init
  E. random_init   — all params, random init (lower bound)

Usage:
    cd /root/rivermind-data/new
    python run_novel_adapt_strategy_exp.py
    python run_novel_adapt_strategy_exp.py --k_shots 20 50 100
"""

import argparse, copy, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import accuracy_score, f1_score

_THIS       = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR  = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
_TRIFACTOR1 = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR_1"))
for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.full_model import FactorizedHARModel
from models.heads import ClassificationHead, ProtoHead
from datasets.base_dataset import POSITION_MAP, SENSOR_TYPE_MAP, discretize_sr

# ── DSADS novel class definitions (same as run_dsads_novel_kshot_v2.py) ───────
DSADS_NOVEL_INDICES = [2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]
NOVEL_CLASS_NAMES = [
    "lying_back", "lying_right", "standing_in_elevator", "moving_in_elevator",
    "walking_parking_lot", "walking_treadmill_inclined", "running_treadmill_8kmh",
    "exercising_stepper", "exercising_cross_trainer",
    "cycling_bike_horizontal", "cycling_bike_vertical",
    "rowing", "jumping", "playing_basketball",
]
NUM_NOVEL = len(DSADS_NOVEL_INDICES)

DSADS_CACHE = os.path.join(
    _TRIFACTOR1, "Cross_person", "_cp_DSADS_windows_w120_h60_c19.npz"
)
_DSADS_META = torch.tensor(
    [0, discretize_sr(20), POSITION_MAP["chest"], SENSOR_TYPE_MAP["acc+gyro"]],
    dtype=torch.long,
)
PRETRAIN_CKPT = os.path.join(
    _THIS, "outputs", "crosshar_exp", "pamap2_novel", "pretrain", "best_model.pth"
)

# ── Dataset ───────────────────────────────────────────────────────────────────
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
        self.signals = sig
        self.labels  = np.array([orig_to_new[int(l)] for l in lab], dtype=np.int64)
        self.subject_ids = subject_ids[mask].astype(np.int64)
        self.class_to_indices = {
            i: np.where(self.labels == i)[0] for i in range(NUM_NOVEL)
        }
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

# ── Adapter ───────────────────────────────────────────────────────────────────
class SemanticAdapter(nn.Module):
    def __init__(self, dim_s, bottleneck=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_s, bottleneck), nn.GELU(),
                                 nn.Linear(bottleneck, dim_s))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
    def forward(self, z): return z + self.net(z)

# ── Strategy setup ────────────────────────────────────────────────────────────
def build_model(cfg, ckpt_path, strategy, device):
    pcfg = copy.deepcopy(cfg)
    pcfg["data"]["num_classes"] = NUM_NOVEL
    pcfg["data"]["num_domains"] = 5
    model = FactorizedHARModel(pcfg)

    # Load pretrain weights (skip for random_init)
    if strategy != "random_init" and os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        cur   = model.state_dict()
        # Only load backbone weights; cls_head shape will differ (14 vs 4 classes)
        filt  = {k: v for k, v in state.items()
                 if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)

    # Replace cls_head for NUM_NOVEL classes (always fresh)
    dim_s     = cfg["model"]["dim_s"]
    cls_cfg   = cfg["model"]["cls_head"]
    proto_cfg = cfg["model"]["proto_head"]
    model.cls_head   = ClassificationHead(dim_s, NUM_NOVEL,
                                          hidden_dim=cls_cfg["hidden_dim"],
                                          dropout=cls_cfg["dropout"])
    model.proto_head = ProtoHead(dim_s, NUM_NOVEL, temperature=proto_cfg["temperature"])

    adapter = None
    if strategy == "linear_probe":
        for p in model.parameters(): p.requires_grad = False
        for p in model.cls_head.parameters(): p.requires_grad = True
    elif strategy == "adapter":
        for p in model.parameters(): p.requires_grad = False
        for p in model.cls_head.parameters(): p.requires_grad = True
        adapter = SemanticAdapter(dim_s, bottleneck=32)
    elif strategy == "partial":
        for p in model.parameters(): p.requires_grad = False
        for p in model.semantic_encoder.parameters(): p.requires_grad = True
        for p in model.cls_head.parameters(): p.requires_grad = True
    else:  # full_ft or random_init
        for p in model.parameters(): p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if adapter is not None:
        n_train += sum(p.numel() for p in adapter.parameters())

    model = model.to(device)
    if adapter is not None: adapter = adapter.to(device)
    return model, adapter, n_train

# ── Training helpers ──────────────────────────────────────────────────────────
def make_loader(ds, bs, shuffle, nw=4):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=nw, pin_memory=True)

def train_epoch(model, adapter, loader, optimizer, criterion, device, grad_clip):
    model.train()
    if adapter is not None: adapter.train()
    total_loss, n_corr, n = 0., 0, 0
    for batch in loader:
        x, m, y = (batch["x_time"].to(device),
                   batch["meta"].to(device),
                   batch["y"].to(device))
        optimizer.zero_grad()
        if adapter is not None:
            with torch.no_grad():
                enc = model.shared_encoder(x)
                z_base = model.semantic_encoder(enc["h_shared"])
            logits = model.cls_head(adapter(z_base))
        else:
            logits = model(x, m)["logits_main"]
        loss = criterion(logits, y)
        loss.backward()
        all_p = [p for p in model.parameters() if p.requires_grad]
        if adapter is not None: all_p += list(adapter.parameters())
        nn.utils.clip_grad_norm_(all_p, grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n_corr     += (logits.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss / max(n, 1), n_corr / max(n, 1)

@torch.no_grad()
def evaluate(model, adapter, loader, device):
    model.eval()
    if adapter is not None: adapter.eval()
    yt, yp = [], []
    for batch in loader:
        x, m = batch["x_time"].to(device), batch["meta"].to(device)
        if adapter is not None:
            enc = model.shared_encoder(x)
            z   = adapter(model.semantic_encoder(enc["h_shared"]))
            preds = model.cls_head(z).argmax(1).cpu()
        else:
            preds = model(x, m)["logits_main"].argmax(1).cpu()
        yt.extend(batch["y"].tolist()); yp.extend(preds.tolist())
    return np.array(yt), np.array(yp)

def run_finetune(model, adapter, ft_loader, val_loader, cfg,
                 device, strategy, out_dir):
    ft_cfg    = cfg.get("finetune", {})
    t_cfg     = cfg["training"]
    # linear_probe / adapter need more epochs; full_ft converges faster
    if strategy in ("linear_probe", "adapter"):
        epochs, lr = 200, 1e-3
    elif strategy == "partial":
        epochs = (ft_cfg.get("stage2_epochs", 20)
                + ft_cfg.get("stage3_epochs", 15)
                + ft_cfg.get("stage4_epochs", 15))
        lr = float(ft_cfg.get("lr", t_cfg["optimizer"]["lr"]))
    else:
        epochs = (ft_cfg.get("stage2_epochs", 20)
                + ft_cfg.get("stage3_epochs", 15)
                + ft_cfg.get("stage4_epochs", 15))
        lr = float(ft_cfg.get("lr", t_cfg["optimizer"]["lr"]))

    wd        = float(t_cfg["optimizer"]["weight_decay"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    min_lr    = float(t_cfg["scheduler"].get("min_lr", 1e-5))
    warmup    = min(5, epochs // 10)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if adapter is not None: trainable += list(adapter.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(1, epochs - warmup)
        return (min_lr / lr) + (1 - min_lr / lr) * 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(out_dir, exist_ok=True)

    best_f1, best_state, best_ada = 0., None, None
    log_every = max(1, epochs // 10)

    for ep in range(epochs):
        loss, acc = train_epoch(model, adapter, ft_loader, optimizer,
                                criterion, device, grad_clip)
        scheduler.step()
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            yt, yp = evaluate(model, adapter, val_loader, device)
            f1  = f1_score(yt, yp, average="macro", zero_division=0)
            vacc = accuracy_score(yt, yp)
            print(f"  [E{ep+1:4d}] loss={loss:.4f}  tr_acc={acc:.4f}  "
                  f"val_acc={vacc:.4f}  val_f1={f1:.4f}")
            if f1 >= best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if adapter: best_ada = {k: v.cpu().clone()
                                        for k, v in adapter.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    if adapter and best_ada: adapter.load_state_dict(best_ada)
    print(f"  Best val F1={best_f1:.4f}")

# ── Per-class report ──────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, precision_score, recall_score

def per_class_report(y_true, y_pred, class_names, tag=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if tag: print(f"\n  {tag}")
    print(f"  {'Class':<35} | {'Acc':>7} | {'F1':>7} | {'N':>6}")
    print(f"  {'─'*58}")
    per = {}
    for i, name in enumerate(class_names):
        sup = int((y_true == i).sum())
        if sup == 0: continue
        acc_i = cm[i, i] / sup
        p_i   = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        r_i   = recall_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        f1_i  = 2*p_i*r_i/(p_i+r_i) if (p_i+r_i) > 0 else 0.
        print(f"  {name:<35} | {acc_i:>7.4f} | {f1_i:>7.4f} | {sup:>6}")
        per[name] = {"acc": float(acc_i), "f1": float(f1_i), "support": sup}
    oa  = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  {'─'*58}")
    print(f"  {'Macro avg':<35} | {'':>7} | {mf1:>7.4f} | {len(y_true):>6}")
    print(f"  {'Overall acc':<35} | {oa:>7.4f} | {mf1:>7.4f} | {len(y_true):>6}")
    per.update({"_overall_acc": float(oa), "_macro_f1": float(mf1)})
    return per

# ── Main ──────────────────────────────────────────────────────────────────────
STRATEGIES = ["linear_probe", "adapter", "partial", "full_ft", "random_init"]
LABELS = {
    "linear_probe": "A. Linear Probe  (17K params)",
    "adapter":      "B. Adapter       (33K params)",
    "partial":      "C. Partial FT    (166K params)",
    "full_ft":      "D. Full FT       (1.41M, pretrained)",
    "random_init":  "E. Full FT       (1.41M, random init)",
}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=os.path.join(_THIS, "configs", "crosshar_exp.yaml"))
    parser.add_argument("--ckpt",       default=PRETRAIN_CKPT)
    parser.add_argument("--k_shots",    type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed
    set_seed(seed)

    print("\nLoading DSADS novel classes ...")
    novel_ds = load_dsads_novel(cfg["data"].get("instance_norm", True))

    # all_results[strategy][k] = {acc, f1_macro, per_class}
    all_results = {s: {} for s in args.strategies}

    for k in args.k_shots:
        set_seed(seed)
        ft_idx, test_idx = build_kshot_splits(novel_ds, k, seed)
        ft_ds   = Subset(novel_ds, ft_idx)
        test_ds = Subset(novel_ds, test_idx)

        bs_ft  = min(128, max(1, len(ft_ds)))
        nw     = cfg["data"]["num_workers"]
        ft_loader   = make_loader(ft_ds,   bs_ft, shuffle=True,  nw=nw)
        test_loader = make_loader(test_ds, 256,   shuffle=False, nw=nw)

        print(f"\n{'━'*68}")
        print(f"  K = {k}  ({k}×{NUM_NOVEL}={k*NUM_NOVEL} FT | "
              f"{(792-k)*NUM_NOVEL} test)")
        print(f"{'━'*68}")

        for strategy in args.strategies:
            set_seed(seed)
            print(f"\n  ── {LABELS[strategy]} ──")
            model, adapter, n_train = build_model(cfg, args.ckpt, strategy, args.device)
            print(f"  Trainable: {n_train:,}")

            out_dir = os.path.join(
                cfg["experiment"]["output_dir"],
                "novel_adapt_strategy", f"k{k:03d}", strategy,
            )
            run_finetune(model, adapter, ft_loader, test_loader,
                         cfg, args.device, strategy, out_dir)

            yt, yp = evaluate(model, adapter, test_loader, args.device)
            per = per_class_report(yt, yp, NOVEL_CLASS_NAMES,
                                   tag=f"K={k}  strategy={strategy}")
            all_results[strategy][k] = per
            with open(os.path.join(out_dir, "results.json"), "w") as f:
                json.dump(per, f, indent=2)

    # ── Summary tables ────────────────────────────────────────────────────────
    ks = args.k_shots
    for metric_key, metric_label in [("_macro_f1", "F1-macro"),
                                      ("_overall_acc", "Accuracy")]:
        print(f"\n{'═'*70}")
        print(f"  NOVEL-CLASS ADAPTATION  (DSADS 14 classes)  — {metric_label}")
        print(f"{'═'*70}")
        hdr = f"  {'Strategy':<40}" + "".join(f"  K={k:>4}" for k in ks)
        print(hdr)
        print(f"  {'─'*68}")
        for s in args.strategies:
            row = f"  {LABELS[s]:<40}"
            for k in ks:
                v = all_results[s].get(k, {}).get(metric_key, float("nan"))
                row += f"  {v:>6.4f}"
            print(row)
        print(f"{'═'*70}")

    # Pretrain gain
    if "full_ft" in args.strategies and "random_init" in args.strategies:
        print(f"\n  Pretrain gain  full_ft vs random_init  [F1-macro Δ]")
        print(f"  {'─'*50}")
        for k in ks:
            ft = all_results["full_ft"].get(k, {}).get("_macro_f1", float("nan"))
            ri = all_results["random_init"].get(k, {}).get("_macro_f1", float("nan"))
            print(f"  K={k:>3}: pretrained={ft:.4f}  random={ri:.4f}  Δ=+{ft-ri:.4f}")

    # Per-class F1 table for best strategy at K=100
    best_s = max(args.strategies,
                 key=lambda s: all_results[s].get(max(ks), {}).get("_macro_f1", 0))
    print(f"\n  Per-class F1  (best strategy={LABELS[best_s]}, K={max(ks)})")
    print(f"  {'Class':<35} | {'F1':>7} | {'Acc':>7}")
    print(f"  {'─'*52}")
    for name in NOVEL_CLASS_NAMES:
        v = all_results[best_s].get(max(ks), {}).get(name, {})
        if isinstance(v, dict):
            print(f"  {name:<35} | {v.get('f1', float('nan')):>7.4f} | "
                  f"{v.get('acc', float('nan')):>7.4f}")

    out_path = os.path.join(
        cfg["experiment"]["output_dir"], "novel_adapt_strategy", "summary.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({s: {str(k): v for k, v in res.items()}
                   for s, res in all_results.items()}, f, indent=2)
    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()

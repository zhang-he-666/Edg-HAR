"""
Cross-dataset adaptation strategy comparison experiment.

Compares 4 strategies for adapting a pretrained foundation model
to a new target domain (PAMAP2) using only X% labeled data:

  A. linear_probe   — freeze all backbone, train cls_head only
  B. adapter        — freeze backbone, insert 2-layer MLP adapter
                      after semantic_encoder, train adapter + cls_head
  C. partial        — freeze shared_encoder (TCN+Transformer),
                      fine-tune semantic_encoder + cls_head
  D. full_ft        — all parameters fine-tuned (existing baseline)
  E. random_init    — no pretrain, full_ft from scratch (lower bound)

Target: PAMAP2 (same 4-class label space as pretrain sources)
Rates:  2%, 5%, 10%

Usage:
    cd /root/rivermind-data/new
    python run_adapt_strategy_exp.py
    python run_adapt_strategy_exp.py --rates 0.02 0.05 0.10
"""

import argparse, copy, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import accuracy_score, f1_score

_THIS      = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.full_model import FactorizedHARModel

# ── Data ──────────────────────────────────────────────────────────────────────
def _load_crosshar():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "crosshar_datasets",
        os.path.join(_THIS, "data_loaders", "crosshar_datasets.py"),
    )
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.CROSSHAR_DATASETS, m.load_single_dataset

def build_pamap2_splits(data_root, rate, seed, instance_norm):
    """finetune=rate% of PAMAP2, test=rest. Same split logic as run_crosshar_exp.py."""
    CROSSHAR_DATASETS, load_single_dataset = _load_crosshar()
    rng = np.random.RandomState(seed)
    ds  = load_single_dataset(data_root, 4, instance_norm=instance_norm)  # id=4 = PAMAP2
    n   = len(ds)
    n_ft = max(1, int(n * rate))
    idx  = rng.permutation(n)
    return Subset(ds, idx[:n_ft].tolist()), Subset(ds, idx[n_ft:].tolist())

# ── Adapter module ────────────────────────────────────────────────────────────
class SemanticAdapter(nn.Module):
    """Lightweight 2-layer MLP inserted after semantic_encoder output (dim_s → dim_s)."""
    def __init__(self, dim_s: int, bottleneck: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_s, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim_s),
        )
        # Init as near-identity so pretrained features are preserved at start
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z_s):
        return z_s + self.net(z_s)   # residual

# ── Strategy implementations ──────────────────────────────────────────────────

def apply_strategy(model: FactorizedHARModel, strategy: str, dim_s: int):
    """
    Configure which parameters are trainable and attach adapter if needed.
    Returns (model, adapter_or_None, param_count_trainable).
    """
    adapter = None

    if strategy == "linear_probe":
        # Freeze everything, unfreeze only cls_head
        for p in model.parameters():
            p.requires_grad = False
        for p in model.cls_head.parameters():
            p.requires_grad = True

    elif strategy == "adapter":
        # Freeze everything
        for p in model.parameters():
            p.requires_grad = False
        # Attach adapter (will be returned separately, trained alongside cls_head)
        adapter = SemanticAdapter(dim_s, bottleneck=32)
        for p in model.cls_head.parameters():
            p.requires_grad = True
        # adapter params managed separately

    elif strategy == "partial":
        # Freeze shared_encoder (TCN + Transformer), unfreeze semantic_encoder + cls_head
        for p in model.parameters():
            p.requires_grad = False
        for p in model.semantic_encoder.parameters():
            p.requires_grad = True
        for p in model.cls_head.parameters():
            p.requires_grad = True

    elif strategy in ("full_ft", "random_init"):
        # All params trainable
        for p in model.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if adapter is not None:
        n_train += sum(p.numel() for p in adapter.parameters())
    return model, adapter, n_train


def build_model(cfg, ckpt_path, strategy, device):
    """Build model, load pretrain weights (except random_init), apply strategy."""
    pcfg = copy.deepcopy(cfg)
    pcfg["data"]["num_domains"] = 5

    model = FactorizedHARModel(pcfg)

    if strategy != "random_init" and os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        cur   = model.state_dict()
        filt  = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)

    dim_s = cfg["model"]["dim_s"]
    model, adapter, n_train = apply_strategy(model, strategy, dim_s)
    model = model.to(device)
    if adapter is not None:
        adapter = adapter.to(device)

    return model, adapter, n_train

# ── Training helpers ───────────────────────────────────────────────────────────

def make_loader(ds, bs, shuffle, num_workers=4, drop_last=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last)


def train_epoch(model, adapter, loader, optimizer, criterion, device, grad_clip):
    model.train()
    if adapter is not None: adapter.train()
    total_loss, n_corr, n = 0., 0, 0
    for batch in loader:
        x = batch["x_time"].to(device)
        m = batch["meta"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()

        # Forward: adapter is inserted between semantic_encoder and cls_head
        if adapter is not None:
            with torch.no_grad():
                enc = model.shared_encoder(x)
                z_s_base = model.semantic_encoder(enc["h_shared"])
            z_s = adapter(z_s_base)
            logits = model.cls_head(z_s)
        else:
            out    = model(x, m)
            logits = out["logits_main"]

        loss = criterion(logits, y)
        loss.backward()
        # Clip across all trainable params
        all_params = list(p for p in model.parameters() if p.requires_grad)
        if adapter is not None:
            all_params += list(adapter.parameters())
        nn.utils.clip_grad_norm_(all_params, grad_clip)
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
        x = batch["x_time"].to(device)
        m = batch["meta"].to(device)
        if adapter is not None:
            enc = model.shared_encoder(x)
            z_s = adapter(model.semantic_encoder(enc["h_shared"]))
            preds = model.cls_head(z_s).argmax(1).cpu()
        else:
            out   = model(x, m)
            preds = out["logits_main"].argmax(1).cpu()
        yt.extend(batch["y"].tolist()); yp.extend(preds.tolist())
    return np.array(yt), np.array(yp)


def run_finetune(model, adapter, ft_loader, val_loader, cfg, device, strategy, out_dir):
    ft_cfg    = cfg.get("finetune", {})
    t_cfg     = cfg["training"]
    epochs    = (ft_cfg.get("stage2_epochs", 20)
               + ft_cfg.get("stage3_epochs", 15)
               + ft_cfg.get("stage4_epochs", 15))
    # linear_probe benefits from higher LR; adapter/partial use finetune LR
    if strategy == "linear_probe":
        lr = 1e-3
        epochs = max(epochs, 100)   # needs more epochs with frozen backbone
    elif strategy == "adapter":
        lr = 1e-3
        epochs = max(epochs, 100)
    else:
        lr = float(ft_cfg.get("lr", t_cfg["optimizer"]["lr"]))

    wd        = float(t_cfg["optimizer"]["weight_decay"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    min_lr    = float(t_cfg["scheduler"].get("min_lr", 1e-5))
    warmup    = min(5, epochs // 10)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if adapter is not None:
        trainable += list(adapter.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)

    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(1, epochs - warmup)
        return (min_lr / lr) + (1 - min_lr / lr) * 0.5 * (1 + np.cos(np.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(out_dir, exist_ok=True)

    best_f1, best_state, best_ada_state = 0., None, None
    log_interval = max(1, epochs // 10)

    for ep in range(epochs):
        loss, acc = train_epoch(model, adapter, ft_loader, optimizer,
                                criterion, device, grad_clip)
        scheduler.step()
        if (ep + 1) % log_interval == 0 or ep == epochs - 1:
            yt, yp = evaluate(model, adapter, val_loader, device)
            f1 = f1_score(yt, yp, average="macro", zero_division=0)
            val_acc = accuracy_score(yt, yp)
            print(f"  [E{ep+1:4d}] loss={loss:.4f}  train_acc={acc:.4f}  "
                  f"val_acc={val_acc:.4f}  val_f1={f1:.4f}")
            if f1 >= best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if adapter is not None:
                    best_ada_state = {k: v.cpu().clone()
                                      for k, v in adapter.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    if adapter is not None and best_ada_state:
        adapter.load_state_dict(best_ada_state)
    print(f"  Best val F1={best_f1:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

STRATEGIES = ["linear_probe", "adapter", "partial", "full_ft", "random_init"]

STRATEGY_LABELS = {
    "linear_probe": "A. Linear Probe (frozen backbone)",
    "adapter":      "B. Adapter      (frozen + 2-layer MLP)",
    "partial":      "C. Partial FT   (unfreeze semantic_enc + head)",
    "full_ft":      "D. Full FT      (all params, pretrained init)",
    "random_init":  "E. Full FT      (random init, no pretrain)",
}

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def main():
    PRETRAIN_CKPT = os.path.join(
        _THIS, "outputs", "crosshar_exp", "pamap2_novel", "pretrain", "best_model.pth"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=os.path.join(_THIS, "configs", "crosshar_exp.yaml"))
    parser.add_argument("--ckpt",     default=PRETRAIN_CKPT)
    parser.add_argument("--rates",    type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    seed  = args.seed

    # all_results[strategy][rate] = {acc, f1_macro}
    all_results = {s: {} for s in args.strategies}

    for rate in args.rates:
        set_seed(seed)
        print(f"\n{'━'*65}")
        print(f"  PAMAP2 target  finetune_rate={rate:.0%}")
        print(f"{'━'*65}")

        ft_ds, test_ds = build_pamap2_splits(
            d_cfg["data_root"], rate, seed,
            d_cfg.get("instance_norm", True),
        )
        bs_ft  = min(d_cfg.get("finetune_batch_size", d_cfg["train_batch_size"]),
                     max(1, len(ft_ds)))
        nw     = d_cfg["num_workers"]
        ft_loader   = make_loader(ft_ds,   bs_ft, shuffle=True,  num_workers=nw)
        test_loader = make_loader(test_ds, d_cfg["val_batch_size"], shuffle=False, num_workers=nw)
        print(f"  finetune={len(ft_ds)}  test={len(test_ds)}")

        for strategy in args.strategies:
            set_seed(seed)
            print(f"\n  ── {STRATEGY_LABELS[strategy]} ──")
            model, adapter, n_train = build_model(cfg, args.ckpt, strategy, args.device)
            print(f"  Trainable params: {n_train:,}  ({n_train/1e6:.3f}M)")

            out_dir = os.path.join(
                cfg["experiment"]["output_dir"],
                "adapt_strategy", f"rate{int(rate*100):02d}pct", strategy,
            )
            run_finetune(model, adapter, ft_loader, test_loader,
                         cfg, args.device, strategy, out_dir)

            yt, yp = evaluate(model, adapter, test_loader, args.device)
            acc    = accuracy_score(yt, yp)
            f1     = f1_score(yt, yp, average="macro", zero_division=0)
            print(f"  TEST  Acc={acc:.4f}  F1-macro={f1:.4f}")
            all_results[strategy][rate] = {"acc": acc, "f1_macro": f1,
                                           "n_train_params": n_train}

    # ── Summary table ──────────────────────────────────────────────────────────
    rates = args.rates
    print(f"\n{'='*72}")
    print(f"  ADAPTATION STRATEGY COMPARISON  (PAMAP2 cross-dataset)")
    print(f"  Metric: F1-macro")
    print(f"{'='*72}")
    rate_hdr = "".join(f"  {r:.0%}".rjust(8) for r in rates)
    print(f"  {'Strategy':<42}{rate_hdr}")
    print(f"  {'─'*70}")
    for s in args.strategies:
        row = f"  {STRATEGY_LABELS[s]:<42}"
        for r in rates:
            v = all_results[s].get(r, {}).get("f1_macro", float("nan"))
            row += f"  {v:>6.4f}"
        print(row)
    print(f"{'='*72}")

    print(f"\n  Metric: Accuracy")
    print(f"{'='*72}")
    print(f"  {'Strategy':<42}{rate_hdr}")
    print(f"  {'─'*70}")
    for s in args.strategies:
        row = f"  {STRATEGY_LABELS[s]:<42}"
        for r in rates:
            v = all_results[s].get(r, {}).get("acc", float("nan"))
            row += f"  {v:>6.4f}"
        print(row)
    print(f"{'='*72}")

    # Pretrain gain over random_init
    if "random_init" in args.strategies and "full_ft" in args.strategies:
        print(f"\n  Pretrain gain (full_ft vs random_init)  [F1-macro delta]")
        print(f"  {'─'*50}")
        for r in rates:
            ft  = all_results["full_ft"].get(r, {}).get("f1_macro", float("nan"))
            ri  = all_results["random_init"].get(r, {}).get("f1_macro", float("nan"))
            print(f"  {r:.0%}: full_ft={ft:.4f}  random_init={ri:.4f}  "
                  f"delta=+{ft-ri:.4f}")

    out_path = os.path.join(
        cfg["experiment"]["output_dir"], "adapt_strategy", "summary.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({s: {str(r): v for r, v in res.items()}
                   for s, res in all_results.items()}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

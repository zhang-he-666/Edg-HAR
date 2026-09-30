"""
1-to-3 跨数据集迁移实验

协议：
  - 在单个源数据集（80% 受试者）上预训练
  - 对其余3个目标数据集各取 2% 数据微调（fine-tune）
  - 在目标数据集剩余 98% 上测试，报告 ACC / macro-F1
  - 4个数据集轮流作为源，共12次迁移对

Usage:
    # 所有4个源，默认2% fine-tune
    python run_1to3_transfer_exp.py

    # 单源单目标
    python run_1to3_transfer_exp.py --source UCI_HAR --target HHAR

    # 调试模式
    python run_1to3_transfer_exp.py --quick

    # 指定输出目录与设备
    python run_1to3_transfer_exp.py --output_dir outputs/1to3 --device cuda:1
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

torch.set_float32_matmul_precision("high")

_THIS      = os.path.dirname(os.path.abspath(__file__))
_TRIFACTOR = os.path.normpath(os.path.join(_THIS, "..", "TriFactor-HAR"))
for p in (_TRIFACTOR, _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets.real_datasets import RealHARDataset
from datasets.base_dataset import POSITION_MAP, SENSOR_TYPE_MAP
from models.full_model import FactorizedHARModel
from trainers.pretrain_trainer import PretrainTrainer
from trainers.factorized_trainer import FactorizedTrainer
from utils.metrics import MetricsTracker

# ─── 常量（与 run_new_datasets_exp.py 保持一致）────────────────────────────────

LABEL_WALK      = 0
LABEL_STAIRS_UP = 1
LABEL_STAIRS_DN = 2
LABEL_SIT_STAND = 3
NUM_CLASSES     = 4

TARGET_SR = 20
WIN_SIZE  = 120
HOP_SIZE  = 60

ALL_DATASETS = ["UCI_HAR", "Shoaib", "Motion", "HHAR"]

DATASET_META_NEW = {
    "UCI_HAR": {
        "id": 0,
        "sampling_rate": TARGET_SR,
        "position": POSITION_MAP["waist"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    "Shoaib": {
        "id": 1,
        "sampling_rate": TARGET_SR,
        "position": POSITION_MAP["upper_arm"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    "Motion": {
        "id": 2,
        "sampling_rate": TARGET_SR,
        "position": POSITION_MAP["wrist"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    "HHAR": {
        "id": 3,
        "sampling_rate": TARGET_SR,
        "position": POSITION_MAP["pocket"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
}

SHOAIB_POSITIONS = ["Arm", "Belt", "Pocket", "Wrist"]
SHOAIB_POSITION_META = {
    "Arm":    {**DATASET_META_NEW["Shoaib"], "position": POSITION_MAP["upper_arm"]},
    "Belt":   {**DATASET_META_NEW["Shoaib"], "position": POSITION_MAP["waist"]},
    "Pocket": {**DATASET_META_NEW["Shoaib"], "position": POSITION_MAP["pocket"]},
    "Wrist":  {**DATASET_META_NEW["Shoaib"], "position": POSITION_MAP["wrist"]},
}

# ─── 数据加载（直接复用 run_new_datasets_exp.py 的实现）────────────────────────

from run_new_datasets_exp import (
    get_records,
    load_shoaib_by_position,
    _resample,
    _highpass,
    _window_signal,
    _load_cache,
    _save_cache,
)


# ─── 数据流构建 ────────────────────────────────────────────────────────────────

def build_source_loaders(
    source_name: str,
    data_root: str,
    batch_size: int = 512,
    num_workers: int = 4,
    label_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    在单个源数据集上构建 train / val loader（subject-split）。
    返回 (train_loader, val_loader)。
    """
    rng = np.random.RandomState(seed)

    train_dsets, val_dsets = [], []

    if source_name == "Shoaib":
        by_pos = load_shoaib_by_position(data_root)
        for pos_name, pos_records in by_pos.items():
            if not pos_records:
                continue
            meta = SHOAIB_POSITION_META[pos_name]
            subs = rng.permutation(sorted(set(r[2] for r in pos_records))).tolist()
            n_tr = max(1, int(len(subs) * label_ratio))
            tr_subs, va_subs = set(subs[:n_tr]), set(subs[n_tr:])

            tr_recs = [r for r in pos_records if r[2] in tr_subs]
            va_recs = [r for r in pos_records if r[2] in va_subs]

            tr_ds = RealHARDataset(
                records=tr_recs, window_size=WIN_SIZE, hop_size=HOP_SIZE,
                ds_meta=meta, normalize=True, instance_norm=True,
                augment=True, aug_mode="all",
            )
            tr_ds._tensor_cache = None
            va_ds = RealHARDataset(
                records=va_recs, window_size=WIN_SIZE, hop_size=HOP_SIZE,
                ds_meta=meta, normalize=True, instance_norm=True, augment=False,
            )
            train_dsets.append(tr_ds)
            val_dsets.append(va_ds)
    else:
        records = get_records(source_name, data_root)
        meta    = DATASET_META_NEW[source_name]
        subs    = rng.permutation(sorted(set(r[2] for r in records))).tolist()
        n_tr    = max(1, int(len(subs) * label_ratio))
        tr_subs, va_subs = set(subs[:n_tr]), set(subs[n_tr:])

        tr_recs = [r for r in records if r[2] in tr_subs]
        va_recs = [r for r in records if r[2] in va_subs]

        tr_ds = RealHARDataset(
            records=tr_recs, window_size=WIN_SIZE, hop_size=HOP_SIZE,
            ds_meta=meta, normalize=True, instance_norm=True,
            augment=True, aug_mode="all",
        )
        tr_ds._tensor_cache = None
        va_ds = RealHARDataset(
            records=va_recs, window_size=WIN_SIZE, hop_size=HOP_SIZE,
            ds_meta=meta, normalize=True, instance_norm=True, augment=False,
        )
        train_dsets.append(tr_ds)
        val_dsets.append(va_ds)

    train_loader = DataLoader(
        ConcatDataset(train_dsets),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        ConcatDataset(val_dsets),
        batch_size=batch_size * 2, shuffle=False,
        num_workers=0,
    )
    n_tr_total = sum(len(d) for d in train_dsets)
    n_va_total = sum(len(d) for d in val_dsets)
    print(f"  [{source_name}] pretrain: {n_tr_total} train / {n_va_total} val windows")
    return train_loader, val_loader


def build_target_loaders(
    target_name: str,
    data_root: str,
    ft_rate: float = 0.02,
    batch_size: int = 512,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[Optional[DataLoader], DataLoader]:
    """
    从目标数据集随机抽取 ft_rate 比例的样本作为 fine-tune 集，
    其余作为 test 集。
    返回 (ft_loader, test_loader)。ft_loader 在 ft_rate=0 时为 None。
    """
    rng = np.random.RandomState(seed)
    records = get_records(target_name, data_root)
    meta    = DATASET_META_NEW[target_name]

    # 按 sample index 随机分割（不按 subject，保证 2% 能拿到各类样本）
    idx = rng.permutation(len(records)).tolist()
    n_ft = max(1, int(len(idx) * ft_rate)) if ft_rate > 0 else 0
    ft_idx   = set(idx[:n_ft])
    test_idx = set(idx[n_ft:])

    ft_recs   = [records[i] for i in sorted(ft_idx)]
    test_recs = [records[i] for i in sorted(test_idx)]

    ft_loader = None
    if ft_rate > 0 and len(ft_recs) > 0:
        ft_ds = RealHARDataset(
            records=ft_recs, window_size=WIN_SIZE, hop_size=HOP_SIZE,
            ds_meta=meta, normalize=True, instance_norm=True,
            augment=True, aug_mode="all",
        )
        ft_ds._tensor_cache = None
        ft_bs = min(batch_size, len(ft_ds))
        ft_loader = DataLoader(
            ft_ds, batch_size=ft_bs, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=False,
            persistent_workers=(num_workers > 0),
        )

    test_ds = RealHARDataset(
        records=test_recs, window_size=WIN_SIZE, hop_size=HOP_SIZE,
        ds_meta=meta, normalize=True, instance_norm=True, augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0,
    )

    print(f"  [{target_name}] ft={len(ft_recs)} ({ft_rate:.1%})  "
          f"test={len(test_recs)} ({1-ft_rate:.1%}) windows")
    return ft_loader, test_loader


# ─── 配置构建 ────────────────────────────────────────────────────────────────────

def build_cfg(quick: bool = False) -> dict:
    cfg_dir = os.path.join(_TRIFACTOR, "configs")
    with open(os.path.join(cfg_dir, "train.yaml")) as f:
        cfg = yaml.safe_load(f)

    cfg["data"]["num_classes"]   = NUM_CLASSES
    cfg["data"]["num_domains"]   = 1      # 预训练阶段只有单源域
    cfg["data"]["window_size"]   = WIN_SIZE
    cfg["data"]["hop_size"]      = HOP_SIZE
    cfg["data"]["sampling_rate"] = TARGET_SR

    if quick:
        cfg["training"]["stage1_epochs"] = 2
        cfg["training"]["stage2_epochs"] = 3
        cfg["training"]["stage3_epochs"] = 2
        cfg["training"]["stage4_epochs"] = 3
        cfg["experiment"]["eval_every"]  = 2
        cfg["experiment"]["log_every"]   = 999999
        cfg["experiment"]["save_every"]  = 999
    else:
        cfg["training"]["stage1_epochs"] = 20
        cfg["training"]["stage2_epochs"] = 20
        cfg["training"]["stage3_epochs"] = 20
        cfg["training"]["stage4_epochs"] = 20
        cfg["experiment"]["eval_every"]  = 3
        cfg["experiment"]["log_every"]   = 100
        cfg["experiment"]["save_every"]  = 999

    return cfg


def build_finetune_cfg(base_cfg: dict, quick: bool = False) -> dict:
    """Fine-tune 阶段使用较少 epoch 和较低 LR，避免在 2% 数据上过拟合。"""
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["num_domains"] = 1  # 微调阶段只有目标域

    ft_epochs = 3 if quick else 15
    cfg["training"]["stage1_epochs"] = 0
    cfg["training"]["stage2_epochs"] = ft_epochs
    cfg["training"]["stage3_epochs"] = ft_epochs
    cfg["training"]["stage4_epochs"] = ft_epochs
    cfg["training"]["optimizer"]["lr"] = 1e-4       # 低学习率防止遗忘
    cfg["training"]["scheduler"]["min_lr"] = 1e-6

    cfg["experiment"]["eval_every"]  = 3 if not quick else 2
    cfg["experiment"]["log_every"]   = 999999
    cfg["experiment"]["save_every"]  = 999

    return cfg


# ─── 评估 ────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_loader(model, loader, device) -> dict:
    model.eval()
    tracker = MetricsTracker(NUM_CLASSES)
    for batch in loader:
        x     = batch["x_time"].to(device)
        meta  = batch["meta"].to(device)
        y     = batch["y"].to(device)
        x_fft = batch.get("x_fft")
        if x_fft is not None:
            x_fft = x_fft.to(device)
        out    = model.forward(x, meta, x_fft=x_fft)
        logits = out["logits_main"]
        tracker.update(F.softmax(logits, dim=-1), y)
    return tracker.compute()


# ─── 单次迁移：source → target ───────────────────────────────────────────────────

def run_one_transfer(
    source_name: str,
    target_name: str,
    base_cfg: dict,
    data_root: str,
    device: str,
    output_dir: str,
    ft_rate: float = 0.02,
    quick: bool = False,
    skip_pretrain: bool = False,
    seed: int = 42,
) -> dict:
    t0 = time.time()
    print(f"\n{'='*68}")
    print(f"  Source: {source_name}  →  Target: {target_name}  (ft={ft_rate:.1%})")
    print(f"{'='*68}")

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── 数据 ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_source_loaders(
        source_name, data_root,
        batch_size=base_cfg["data"]["train_batch_size"],
        num_workers=8, seed=seed,
    )
    ft_loader, test_loader = build_target_loaders(
        target_name, data_root,
        ft_rate=ft_rate,
        batch_size=base_cfg["data"]["train_batch_size"],
        num_workers=8, seed=seed,
    )

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model = FactorizedHARModel(base_cfg)

    pt_dir   = os.path.join(output_dir, "pretrain")
    pt_ckpt  = os.path.join(pt_dir, "best_pretrain.pth")
    ft_dir   = os.path.join(output_dir, "factorized")
    tgt_dir  = os.path.join(output_dir, f"finetune_{target_name}")

    # ── Stage 1：预训练 ────────────────────────────────────────────────────────
    if not skip_pretrain:
        print("\n--- Stage 1: Pretrain ---")
        pt = PretrainTrainer(model, base_cfg, device=device, output_dir=pt_dir)
        pt.train(train_loader, val_loader)
    else:
        print(f"\n--- Stage 1: Skipped ---")

    # ── Stage 2-4：Factorized 训练 ────────────────────────────────────────────
    print("\n--- Stage 2-4: Factorized Training ---")
    ft = FactorizedTrainer(model, base_cfg, device=device, output_dir=ft_dir)
    ft.train(train_loader, val_loader, pretrain_ckpt=pt_ckpt)

    best_src_ckpt = os.path.join(ft_dir, "best_model.pth")

    # ── Source 评估（No TTA）────────────────────────────────────────────────────
    print("\n--- Eval: Source Val (No TTA) ---")
    ckpt = torch.load(best_src_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    src_val_m = eval_loader(model, val_loader, device)
    print(f"  Source Val: acc={src_val_m['accuracy']:.4f}  f1={src_val_m['f1_macro']:.4f}")

    # ── Zero-shot 评估（迁移前）────────────────────────────────────────────────
    print("\n--- Eval: Zero-shot on Target (before FT) ---")
    zs_m = eval_loader(model, test_loader, device)
    print(f"  Zero-shot: acc={zs_m['accuracy']:.4f}  f1={zs_m['f1_macro']:.4f}")

    # ── Fine-tune 目标域 2% ────────────────────────────────────────────────────
    ft_m = {"accuracy": float("nan"), "f1_macro": float("nan")}
    if ft_loader is not None and ft_rate > 0:
        print(f"\n--- Fine-tune on {target_name} ({ft_rate:.1%}) ---")
        ft_cfg = build_finetune_cfg(base_cfg, quick=quick)
        # fine-tune 时 val 用 ft_loader 本身（数据量小，防止引入目标域信息泄漏）
        tgt_ft = FactorizedTrainer(model, ft_cfg, device=device, output_dir=tgt_dir)
        tgt_ft.train(ft_loader, val_loader=ft_loader, pretrain_ckpt=best_src_ckpt)
        best_ft_ckpt = os.path.join(tgt_dir, "best_model.pth")
        if os.path.exists(best_ft_ckpt):
            ckpt = torch.load(best_ft_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=False)
            model.to(device)

        print(f"\n--- Eval: After FT on Target ---")
        ft_m = eval_loader(model, test_loader, device)
        print(f"  After FT:  acc={ft_m['accuracy']:.4f}  f1={ft_m['f1_macro']:.4f}")
    else:
        print("\n  [Skip FT] ft_rate=0, zero-shot only.")

    result = {
        "source":      source_name,
        "target":      target_name,
        "ft_rate":     ft_rate,
        "source_val":  src_val_m,
        "zero_shot":   zs_m,
        "after_ft":    ft_m,
        "elapsed_sec": time.time() - t0,
    }
    res_path = os.path.join(output_dir, "result.json")
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {res_path}  ({result['elapsed_sec']/60:.1f} min)")
    return result


# ─── 主函数 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="1-to-3 cross-dataset transfer experiment")
    parser.add_argument("--source", choices=ALL_DATASETS, default=None,
                        help="Source dataset (default: all 4)")
    parser.add_argument("--target", choices=ALL_DATASETS, default=None,
                        help="Target dataset (default: all non-source)")
    parser.add_argument("--ft_rate", type=float, default=0.02,
                        help="Fine-tune ratio on target (default: 0.02)")
    _default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TriFactor-HAR", "data")
    parser.add_argument("--data_root", default=_default_data,
                        help="Path to raw dataset root")
    _default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "1to3")
    parser.add_argument("--output_dir", default=_default_out,
                        help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Debug mode: fewer epochs")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip Stage 1 (use existing checkpoint)")
    args = parser.parse_args()

    sources = [args.source] if args.source else ALL_DATASETS
    all_results = []
    summary_rows = []

    for src in sources:
        targets = [t for t in ALL_DATASETS if t != src]
        if args.target:
            if args.target == src:
                print(f"  [Skip] source==target: {src}")
                continue
            targets = [args.target]

        cfg = build_cfg(quick=args.quick)
        cfg["experiment"]["seed"] = args.seed

        # 预训练 checkpoint 在同一个 source 下共享，只训练一次
        src_base_dir = os.path.join(args.output_dir, f"source_{src}")

        for tgt in targets:
            run_dir = os.path.join(src_base_dir, f"target_{tgt}")
            # 如果该 source 的 factorized ckpt 已存在，后续 target 可复用
            existing_ft_ckpt = os.path.join(src_base_dir, "first_run", "factorized", "best_model.pth")
            skip_pt = args.skip_pretrain or os.path.exists(existing_ft_ckpt)

            # 第一个 target 时完整训练，写到 first_run 目录供后续复用
            first_run_dir = os.path.join(src_base_dir, "first_run")
            if not os.path.exists(os.path.join(first_run_dir, "factorized", "best_model.pth")):
                # 需要完整跑一遍预训练+factorized
                result = run_one_transfer(
                    source_name=src,
                    target_name=tgt,
                    base_cfg=copy.deepcopy(cfg),
                    data_root=args.data_root,
                    device=args.device,
                    output_dir=first_run_dir,
                    ft_rate=args.ft_rate,
                    quick=args.quick,
                    skip_pretrain=args.skip_pretrain,
                    seed=args.seed,
                )
                # 把 first_run 的结果也算作该 target 的结果
                result["source"] = src
                result["target"] = tgt
                # 把 result.json 也写一份到 run_dir
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "result.json"), "w") as f:
                    json.dump(result, f, indent=2, default=str)
            else:
                # 预训练已完成，直接 fine-tune + eval
                result = run_one_transfer(
                    source_name=src,
                    target_name=tgt,
                    base_cfg=copy.deepcopy(cfg),
                    data_root=args.data_root,
                    device=args.device,
                    output_dir=run_dir,
                    ft_rate=args.ft_rate,
                    quick=args.quick,
                    skip_pretrain=True,
                    seed=args.seed,
                )

            all_results.append(result)
            acc  = result["after_ft"].get("accuracy", float("nan"))
            f1   = result["after_ft"].get("f1_macro", float("nan"))
            zs_acc = result["zero_shot"].get("accuracy", float("nan"))
            zs_f1  = result["zero_shot"].get("f1_macro", float("nan"))
            summary_rows.append({
                "source": src, "target": tgt,
                "zs_acc": zs_acc, "zs_f1": zs_f1,
                "ft_acc": acc,   "ft_f1": f1,
            })
            print(f"\n  ✓ {src}→{tgt}  ZS acc={zs_acc:.4f} f1={zs_f1:.4f}  "
                  f"FT acc={acc:.4f} f1={f1:.4f}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"rows": summary_rows, "all_results": all_results}, f, indent=2, default=str)
    print(f"\n{'='*68}")
    print(f"Summary saved → {summary_path}")
    print(f"\n{'Source':<12} {'Target':<12} {'ZS-ACC':>8} {'ZS-F1':>8} {'FT-ACC':>8} {'FT-F1':>8}")
    print("-" * 60)
    for row in summary_rows:
        print(f"{row['source']:<12} {row['target']:<12} "
              f"{row['zs_acc']:>8.4f} {row['zs_f1']:>8.4f} "
              f"{row['ft_acc']:>8.4f} {row['ft_f1']:>8.4f}")

    # 全局平均
    valid = [r for r in summary_rows if not (
        isinstance(r["ft_acc"], float) and r["ft_acc"] != r["ft_acc"]  # nan check
    )]
    if valid:
        avg_zs_acc = np.mean([r["zs_acc"] for r in valid])
        avg_zs_f1  = np.mean([r["zs_f1"]  for r in valid])
        avg_ft_acc = np.mean([r["ft_acc"] for r in valid])
        avg_ft_f1  = np.mean([r["ft_f1"]  for r in valid])
        print("-" * 60)
        print(f"{'MEAN':<12} {'---':<12} "
              f"{avg_zs_acc:>8.4f} {avg_zs_f1:>8.4f} "
              f"{avg_ft_acc:>8.4f} {avg_ft_f1:>8.4f}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()

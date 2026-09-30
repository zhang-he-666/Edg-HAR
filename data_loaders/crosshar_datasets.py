"""
CrossHAR-style dataset loader for Edge-HAR.

Uses the pre-built _new_*_windows_w120_h60_c4.npz caches.
Each npz has: signals [N,120,6], labels [N] (0-3), subject_ids [N].

Labels: 0=walk, 1=upstairs, 2=downstairs, 3=sit+stand  (4 classes)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset

# ── Make Edge-HAR importable ─────────────────────────────────────────────

_TRIFACTOR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "TriFactor-HAR"))
if _TRIFACTOR not in sys.path:
    sys.path.insert(0, _TRIFACTOR)

from datasets.base_dataset import POSITION_MAP, SENSOR_TYPE_MAP, SensorMeta, discretize_sr


CROSSHAR_DATASETS: Dict[int, dict] = {
    0: {
        "name": "HHAR",
        "file": "_new_HHAR_windows_w120_h60_c4.npz",
        "sampling_rate": 20,
        "position": POSITION_MAP["waist"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    1: {
        "name": "Motion",
        "file": "_new_Motion_windows_w120_h60_c4.npz",
        "sampling_rate": 50,
        "position": POSITION_MAP["wrist"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    2: {
        "name": "UCI_HAR",
        "file": "_new_UCI_HAR_windows_w120_h60_c4.npz",
        "sampling_rate": 50,
        "position": POSITION_MAP["waist"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    3: {
        "name": "Shoaib",
        "file": "_new_Shoaib_windows_w120_h60_c4.npz",
        "sampling_rate": 20,
        "position": POSITION_MAP["pocket"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
    4: {
        "name": "PAMAP2",
        "file": "_new_PAMAP2_windows_w120_h60_c4.npz",
        # Chest IMU @100Hz downsampled to 20Hz in _mc_ cache; treat as 100Hz raw
        "sampling_rate": 100,
        "position": POSITION_MAP["chest"],
        "sensor_type": SENSOR_TYPE_MAP["acc+gyro"],
    },
}

NUM_CLASSES = 4
ACTIVITY_NAMES = ["walking", "upstairs", "downstairs", "sitting&standing"]


# ── Dataset class ─────────────────────────────────────────────────────────────

class CrossHARSingleDataset(Dataset):
    """
    Wraps one npz cache file.  Returns batches compatible with
    Edge-HAR's FactorizedHARModel (x_time, meta, y, domain_id, subject_id).
    """

    def __init__(
        self,
        data_root: str,
        dataset_id: int,
        signals: np.ndarray,      # [N, 120, 6]
        labels: np.ndarray,       # [N]  int
        subject_ids: np.ndarray,  # [N]  int
        instance_norm: bool = True,
    ):
        self.dataset_id = dataset_id
        ds_info = CROSSHAR_DATASETS[dataset_id]
        self.ds_info = ds_info

        # Instance normalisation (same as CrossHAR)
        signals = signals.astype(np.float32)
        if instance_norm:
            data_t = torch.from_numpy(signals).permute(0, 2, 1)  # [N, 6, 120]
            norm = torch.nn.InstanceNorm1d(6, affine=False)
            data_t = norm(data_t)
            signals = data_t.permute(0, 2, 1).numpy()  # [N, 120, 6]

        self.signals = signals
        self.labels = labels.astype(np.int64)
        self.subject_ids = subject_ids.astype(np.int64)

        sr_bin = discretize_sr(ds_info["sampling_rate"])
        self._meta_t = torch.tensor(
            [dataset_id, sr_bin, ds_info["position"], ds_info["sensor_type"]],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.signals[idx]).float()   # [120, 6]
        x_time = x.permute(1, 0)                          # [6, 120]
        x_fft = torch.from_numpy(
            np.abs(np.fft.rfft(self.signals[idx], axis=0)).astype(np.float32)
        ).permute(1, 0)                                    # [6, F]

        return {
            "x":          x,
            "x_time":     x_time,
            "x_fft":      x_fft,
            "y":          torch.tensor(self.labels[idx], dtype=torch.long),
            "domain_id":  torch.tensor(self.dataset_id, dtype=torch.long),
            "subject_id": torch.tensor(int(self.subject_ids[idx]), dtype=torch.long),
            "meta":       self._meta_t,
        }

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_single_dataset(data_root: str, dataset_id: int, instance_norm: bool = True) -> CrossHARSingleDataset:
    ds_info = CROSSHAR_DATASETS[dataset_id]
    path = os.path.join(data_root, ds_info["file"])
    cached = np.load(path)
    signals    = cached["signals"]      # [N, 120, 6]
    labels     = cached["labels"]       # [N]
    subject_ids = cached["subject_ids"] # [N]
    print(f"  [{ds_info['name']}] {len(signals)} windows, "
          f"{len(np.unique(subject_ids))} subjects, "
          f"labels {sorted(np.unique(labels).tolist())}")
    return CrossHARSingleDataset(
        data_root=data_root,
        dataset_id=dataset_id,
        signals=signals,
        labels=labels,
        subject_ids=subject_ids,
        instance_norm=instance_norm,
    )


def build_crosshar_splits(
    data_root: str,
    target_id: int,
    pretrain_source_rate: float = 0.8,
    finetune_target_rate: float = 0.1,
    seed: int = 42,
    instance_norm: bool = True,
) -> Tuple[ConcatDataset, "SubsetDataset", "SubsetDataset"]:
    """
    Returns (pretrain_ds, finetune_ds, test_ds).

    pretrain_ds  — 80% of each SOURCE dataset (all labels but used unsupervised)
    finetune_ds  — 10% of TARGET dataset (labeled)
    test_ds      — remaining TARGET dataset (labeled)

    val split comes from the remaining 20% of source datasets (not returned
    separately; caller may slice pretrain_ds or use test_ds for reporting).
    """
    rng = np.random.RandomState(seed)
    source_ids = [i for i in CROSSHAR_DATASETS if i != target_id]

    # ── Source datasets ───────────────────────────────────────────────────────
    pretrain_parts: List[Dataset] = []
    for src_id in source_ids:
        ds = load_single_dataset(data_root, src_id, instance_norm=instance_norm)
        n = len(ds)
        n_pretrain = int(n * pretrain_source_rate)
        idx = rng.permutation(n)
        pretrain_parts.append(Subset(ds, idx[:n_pretrain].tolist()))
        print(f"  Source {CROSSHAR_DATASETS[src_id]['name']}: "
              f"{n_pretrain}/{n} for pretrain")

    pretrain_ds = ConcatDataset(pretrain_parts)

    # ── Target dataset ────────────────────────────────────────────────────────
    tgt_ds = load_single_dataset(data_root, target_id, instance_norm=instance_norm)
    n_tgt = len(tgt_ds)
    n_finetune = int(n_tgt * finetune_target_rate)
    idx_tgt = rng.permutation(n_tgt)
    finetune_ds = Subset(tgt_ds, idx_tgt[:n_finetune].tolist())
    test_ds     = Subset(tgt_ds, idx_tgt[n_finetune:].tolist())

    print(f"  Target {CROSSHAR_DATASETS[target_id]['name']}: "
          f"{n_finetune} finetune / {n_tgt - n_finetune} test")

    return pretrain_ds, finetune_ds, test_ds

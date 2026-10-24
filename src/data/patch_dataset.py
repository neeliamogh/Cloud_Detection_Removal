from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import read_json


def robust_scale(arr: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    low = low[:, None, None]
    high = high[:, None, None]
    arr = (arr - low) / np.maximum(high - low, 1e-6)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


class CloudPatchDataset(Dataset):
    def __init__(self, split_json: str | Path, stats_json: str | Path):
        self.items = read_json(split_json)
        self.stats = read_json(stats_json)
        self.s2_low = np.array(self.stats["s2_low"], dtype=np.float32)
        self.s2_high = np.array(self.stats["s2_high"], dtype=np.float32)
        self.s1_low = np.array(self.stats["s1_low"], dtype=np.float32)
        self.s1_high = np.array(self.stats["s1_high"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        meta = self.items[idx]
        data = np.load(meta["path"])
        cloudy_s2 = robust_scale(data["cloudy_s2"], self.s2_low, self.s2_high)
        clear_s2 = robust_scale(data["clear_s2"], self.s2_low, self.s2_high)
        s1 = robust_scale(data["s1"], self.s1_low, self.s1_high)
        mask = data["cloud_mask"].astype(np.float32)

        detector_x = cloudy_s2
        remover_x = np.concatenate([cloudy_s2, s1, mask], axis=0).astype(np.float32)
        return {
            "detector_x": torch.from_numpy(detector_x),
            "remover_x": torch.from_numpy(remover_x),
            "target": torch.from_numpy(clear_s2),
            "mask": torch.from_numpy(mask),
            "meta_path": meta["path"],
            "regime": meta["regime"],
        }


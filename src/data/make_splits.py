from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils import load_config, read_json, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["project"]["seed"]))

    manifest = read_json(Path(cfg["paths"]["data_root"]) / "manifest.json")
    holdouts = set(cfg["validation"]["holdout_regimes"])
    by_regime = defaultdict(list)
    for item in manifest:
        by_regime[item["regime"]].append(item)

    train, val, test, cross_regime = [], [], [], []
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    for regime, items in by_regime.items():
        items = list(items)
        rng.shuffle(items)
        if regime in holdouts:
            cross_regime.extend(items)
            continue
        n = len(items)
        if n < 3:
            train.extend(items)
            continue
        n_train = max(1, int(0.70 * n))
        n_val = max(1, int(0.15 * n))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    if not val and train:
        val.append(train.pop())
    if not test and train:
        test.append(train.pop())
    if not cross_regime and test:
        cross_regime.append(test[-1])

    split_dir = Path(cfg["paths"]["split_dir"])
    save_json(train, split_dir / "train.json")
    save_json(val, split_dir / "val.json")
    save_json(test, split_dir / "test.json")
    save_json(cross_regime, split_dir / "cross_regime.json")
    print(f"train={len(train)} val={len(val)} test={len(test)} cross_regime={len(cross_regime)}")


if __name__ == "__main__":
    main()

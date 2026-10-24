from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import load_config, read_json, save_json


def collect_percentiles(paths: list[str], key: str, percentiles: list[int], max_files: int = 1200):
    vals = []
    for path in tqdm(paths[:max_files], desc=f"stats {key}"):
        arr = np.load(path)[key].astype(np.float32)
        vals.append(np.percentile(arr.reshape(arr.shape[0], -1), percentiles, axis=1))
    stacked = np.stack(vals, axis=0)
    lo = np.median(stacked[:, 0, :], axis=0)
    hi = np.median(stacked[:, 1, :], axis=0)
    return lo.tolist(), hi.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = read_json(Path(cfg["paths"]["data_root"]) / "manifest.json")
    paths = [x["path"] for x in manifest]
    if not paths:
        raise RuntimeError("No patches found. Run build_india_stac_dataset first.")

    percentiles = cfg["data"]["normalize_percentiles"]
    s2_lo, s2_hi = collect_percentiles(paths, "clear_s2", percentiles)
    s1_lo, s1_hi = collect_percentiles(paths, "s1", percentiles)
    stats = {
        "s2_low": s2_lo,
        "s2_high": s2_hi,
        "s1_low": s1_lo,
        "s1_high": s1_hi,
        "method": "median_of_per_patch_percentiles",
        "percentiles": percentiles,
    }
    out = Path(cfg["paths"]["data_root"]) / "normalization_stats.json"
    save_json(stats, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()


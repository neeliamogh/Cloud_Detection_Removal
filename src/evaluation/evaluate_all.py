from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.patch_dataset import CloudPatchDataset
from src.evaluation.metrics import f1_score, iou_score, mae, ndvi_error, psnr, rmse, sam
from src.models.detector import build_cloud_detector
from src.models.remover import MultiModalUNet
from src.utils import load_config, save_json


def get_bucket(fraction: float) -> str:
    if fraction <= 0.3:
        return "low"
    elif fraction <= 0.6:
        return "medium"
    elif fraction <= 0.9:
        return "high"
    return "very_high"

def init_totals():
    return {
        "det_iou": 0.0, "det_f1": 0.0, "det_precision": 0.0, "det_recall": 0.0,
        "psnr": 0.0, "mae": 0.0, "rmse": 0.0, "sam": 0.0, "ndvi_error": 0.0,
        "cloud_region_mae": 0.0, "samples": 0
    }

@torch.no_grad()
def evaluate_split(name: str, split_path: Path, cfg: dict, device):
    stats = Path(cfg["paths"]["data_root"]) / "normalization_stats.json"
    ds = CloudPatchDataset(split_path, stats)
    if len(ds) == 0:
        return {"split": name, "samples": 0}
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=int(cfg["training"]["num_workers"]))

    detector = build_cloud_detector(
        in_channels=len(cfg["data"]["sentinel2_bands"]),
        encoder=cfg["training"]["detector"]["encoder"],
    ).to(device)
    detector.load_state_dict(torch.load(Path(cfg["paths"]["model_dir"]) / "best_cloud_detector.pt", map_location=device))
    detector.eval()

    remover = MultiModalUNet(
        in_channels=len(cfg["data"]["sentinel2_bands"]) + len(cfg["data"]["sentinel1_bands"]) + 1,
        out_channels=len(cfg["data"]["sentinel2_bands"]),
    ).to(device)
    remover.load_state_dict(torch.load(Path(cfg["paths"]["model_dir"]) / "best_cloud_remover.pt", map_location=device))
    remover.eval()

    totals = init_totals()
    bucket_totals = {b: init_totals() for b in ["low", "medium", "high", "very_high"]}
    regime_totals = {}

    for batch in tqdm(loader, desc=f"eval {name}"):
        det_x = batch["detector_x"].to(device)
        rem_x = batch["remover_x"].to(device)
        y = batch["target"].to(device)
        mask = batch["mask"].to(device)
        regimes = batch["regime"]
        logits = detector(det_x)
        pred = remover(rem_x)

        for i in range(len(det_x)):
            m_i = mask[i:i+1]
            frac = m_i.mean().item()
            b = get_bucket(frac)
            regime = regimes[i]
            if regime not in regime_totals:
                regime_totals[regime] = init_totals()

            l_i = logits[i:i+1]
            p_i = pred[i:i+1]
            y_i = y[i:i+1]
            det_x_i = det_x[i:i+1]

            f1, precision, recall = f1_score(l_i, m_i)
            c_mae = (torch.abs(p_i - y_i) * m_i).sum() / (m_i.sum() * p_i.shape[1] + 1e-6)
            
            mets = {
                "det_iou": float(iou_score(l_i, m_i).item()),
                "det_f1": float(f1.item()),
                "det_precision": float(precision.item()),
                "det_recall": float(recall.item()),
                "psnr": float(psnr(p_i, y_i).item()),
                "mae": float(mae(p_i, y_i).item()),
                "rmse": float(rmse(p_i, y_i).item()),
                "sam": float(sam(p_i, y_i).item()),
                "ndvi_error": float(ndvi_error(p_i, y_i).item()),
                "cloud_region_mae": float(c_mae.item())
            }

            for k, v in mets.items():
                totals[k] += v
                bucket_totals[b][k] += v
                regime_totals[regime][k] += v
            totals["samples"] += 1
            bucket_totals[b]["samples"] += 1
            regime_totals[regime]["samples"] += 1

    res = {"split": name, "samples": totals["samples"]}
    for k in totals.keys():
        if k != "samples":
            res[k] = totals[k] / max(totals["samples"], 1)
    
    for b, b_tot in bucket_totals.items():
        for k in b_tot.keys():
            if k != "samples":
                res[f"{b}_{k}"] = b_tot[k] / max(b_tot["samples"], 1)
        res[f"{b}_samples"] = b_tot["samples"]

    for reg, reg_tot in regime_totals.items():
        for k in reg_tot.keys():
            if k != "samples":
                res[f"regime_{reg}_{k}"] = reg_tot[k] / max(reg_tot["samples"], 1)
        res[f"regime_{reg}_samples"] = reg_tot["samples"]

    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = Path(cfg["paths"]["split_dir"])
    rows = []
    for name in ["val", "test", "cross_regime"]:
        rows.append(evaluate_split(name, split_dir / f"{name}.json", cfg, device))
    save_json(rows, Path(cfg["paths"]["result_dir"]) / "evaluation_summary.json")
    print(rows)


if __name__ == "__main__":
    main()


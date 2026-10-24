from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.patch_dataset import CloudPatchDataset
from src.evaluation.metrics import mae, ndvi_error, psnr, rmse, sam
from src.models.remover import MultiModalUNet
from src.training.losses import SSIMLoss, cloud_weighted_l1, spectral_angle_loss
from src.utils import ensure_dirs, load_config, save_json, seed_everything


def run_epoch(model, loader, optimizer, scaler, device, cfg, train: bool):
    model.train(train)
    ssim_loss = SSIMLoss()
    totals = {"loss": 0.0, "psnr": 0.0, "mae": 0.0, "rmse": 0.0, "sam": 0.0, "ndvi_error": 0.0}
    for batch in tqdm(loader, desc="train" if train else "val"):
        x = batch["remover_x"].to(device)
        y = batch["target"].to(device)
        mask = batch["mask"].to(device)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred = model(x)
            loss = (
                float(cfg["training"]["remover"]["lambda_l1"]) * cloud_weighted_l1(pred, y, mask, float(cfg["training"]["remover"]["lambda_cloud"]))
                + float(cfg["training"]["remover"]["lambda_ssim"]) * ssim_loss(pred, y)
                + float(cfg["training"]["remover"]["lambda_sam"]) * spectral_angle_loss(pred, y)
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            totals["loss"] += float(loss.item())
            totals["psnr"] += float(psnr(pred, y).item())
            totals["mae"] += float(mae(pred, y).item())
            totals["rmse"] += float(rmse(pred, y).item())
            totals["sam"] += float(sam(pred, y).item())
            totals["ndvi_error"] += float(ndvi_error(pred, y).item())
    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(int(cfg["project"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = Path(cfg["paths"]["data_root"]) / "normalization_stats.json"
    train_ds = CloudPatchDataset(Path(cfg["paths"]["split_dir"]) / "train.json", stats)
    val_ds = CloudPatchDataset(Path(cfg["paths"]["split_dir"]) / "val.json", stats)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["remover"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["remover"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )

    in_channels = len(cfg["data"]["sentinel2_bands"]) + len(cfg["data"]["sentinel1_bands"]) + 1
    model = MultiModalUNet(in_channels=in_channels, out_channels=len(cfg["data"]["sentinel2_bands"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["remover"]["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["remover"]["epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_psnr = -1.0
    history = []

    for epoch in range(1, int(cfg["training"]["remover"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, cfg, True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, cfg, False)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(row)
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(model.state_dict(), Path(cfg["paths"]["model_dir"]) / "best_cloud_remover.pt")
            print(f"Saved best remover psnr={best_psnr:.4f}")

    save_json(history, Path(cfg["paths"]["result_dir"]) / "remover_history.json")


if __name__ == "__main__":
    main()


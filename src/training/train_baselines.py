from __future__ import annotations

import argparse
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.patch_dataset import CloudPatchDataset
from src.evaluation.metrics import mae, ndvi_error, psnr, rmse, sam
from src.training.losses import SSIMLoss, cloud_weighted_l1, spectral_angle_loss
from src.utils import ensure_dirs, load_config, save_json, seed_everything


def evaluate_baselines(cfg, device):
    stats = Path(cfg["paths"]["data_root"]) / "normalization_stats.json"
    train_ds = CloudPatchDataset(Path(cfg["paths"]["split_dir"]) / "train.json", stats)
    test_ds = CloudPatchDataset(Path(cfg["paths"]["split_dir"]) / "test.json", stats)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return

    # Compute training mean per band for mean-pixel baseline
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=int(cfg["training"]["num_workers"]))
    totals = None
    count = 0
    print("Computing train set mean...")
    for batch in tqdm(train_loader):
        y = batch["target"]
        batch_sum = y.sum(dim=(0, 2, 3))
        if totals is None:
            totals = batch_sum
        else:
            totals += batch_sum
        count += y.shape[0] * y.shape[2] * y.shape[3]
    train_mean = (totals / count).to(device)

    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=int(cfg["training"]["num_workers"]))
    
    mean_mets = {"psnr": 0.0, "mae": 0.0, "rmse": 0.0, "sam": 0.0, "ndvi_error": 0.0}
    copy_mets = {"psnr": 0.0, "mae": 0.0, "rmse": 0.0, "sam": 0.0, "ndvi_error": 0.0}

    print("Evaluating naive baselines...")
    for batch in tqdm(test_loader):
        x = batch["detector_x"].to(device)  # S2 cloudy
        y = batch["target"].to(device)
        mask = batch["mask"].to(device)
        
        # Mean pixel prediction
        B, C, H, W = y.shape
        pred_mean = train_mean.view(1, C, 1, 1).expand(B, C, H, W)
        
        # Copy-cloudy prediction
        pred_copy = x
        
        with torch.no_grad():
            mean_mets["psnr"] += float(psnr(pred_mean, y).item())
            mean_mets["mae"] += float(mae(pred_mean, y).item())
            mean_mets["rmse"] += float(rmse(pred_mean, y).item())
            mean_mets["sam"] += float(sam(pred_mean, y).item())
            mean_mets["ndvi_error"] += float(ndvi_error(pred_mean, y).item())
            
            copy_mets["psnr"] += float(psnr(pred_copy, y).item())
            copy_mets["mae"] += float(mae(pred_copy, y).item())
            copy_mets["rmse"] += float(rmse(pred_copy, y).item())
            copy_mets["sam"] += float(sam(pred_copy, y).item())
            copy_mets["ndvi_error"] += float(ndvi_error(pred_copy, y).item())

    n = max(len(test_loader), 1)
    results = {
        "mean_pixel": {k: v / n for k, v in mean_mets.items()},
        "copy_cloudy": {k: v / n for k, v in copy_mets.items()}
    }
    save_json(results, Path(cfg["paths"]["result_dir"]) / "naive_baselines.json")
    print("Naive baselines:", results)

def run_optical_epoch(model, loader, optimizer, scheduler, scaler, device, cfg, train: bool):
    model.train(train)
    ssim_loss = SSIMLoss()
    totals = {"loss": 0.0, "psnr": 0.0, "mae": 0.0, "rmse": 0.0, "sam": 0.0, "ndvi_error": 0.0}
    for batch in tqdm(loader, desc="train" if train else "val"):
        x = batch["detector_x"].to(device)  # Only optical
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
    
    if train and scheduler is not None:
        scheduler.step()

    n = max(len(loader), 1)
    return {k: v / n for k, v in totals.items()}

def train_optical_baseline(cfg, device):
    stats = Path(cfg["paths"]["data_root"]) / "normalization_stats.json"
    train_ds = CloudPatchDataset(Path(cfg["paths"]["split_dir"]) / "train.json", stats)
    val_ds = CloudPatchDataset(Path(cfg["paths"]["split_dir"]) / "val.json", stats)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["training"]["remover"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]), pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["training"]["remover"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]), pin_memory=True
    )

    in_channels = len(cfg["data"]["sentinel2_bands"])
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=in_channels,
        activation="sigmoid",
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["remover"]["lr"]), weight_decay=1e-4)
    epochs = int(cfg["training"]["remover"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_psnr = -1.0
    history = []

    print("Training optical-only baseline...")
    for epoch in range(1, epochs + 1):
        train_metrics = run_optical_epoch(model, train_loader, optimizer, None, scaler, device, cfg, True)
        val_metrics = run_optical_epoch(model, val_loader, optimizer, None, scaler, device, cfg, False)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(row)
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save(model.state_dict(), Path(cfg["paths"]["model_dir"]) / "best_optical_baseline.pt")
            print(f"Saved best optical baseline psnr={best_psnr:.4f}")

    save_json(history, Path(cfg["paths"]["result_dir"]) / "optical_baseline_history.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(int(cfg["project"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_baselines(cfg, device)
    train_optical_baseline(cfg, device)


if __name__ == "__main__":
    main()

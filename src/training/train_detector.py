from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.patch_dataset import CloudPatchDataset
from src.evaluation.metrics import f1_score, iou_score
from src.models.detector import build_cloud_detector
from src.utils import ensure_dirs, load_config, save_json, seed_everything


def run_epoch(model, loader, optimizer, scaler, device, train: bool):
    model.train(train)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, total_iou = 0.0, 0.0
    total_f1, total_precision, total_recall = 0.0, 0.0, 0.0
    for batch in tqdm(loader, desc="train" if train else "val"):
        x = batch["detector_x"].to(device)
        y = batch["mask"].to(device)
        with torch.set_grad_enabled(train), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            f1, precision, recall = f1_score(logits, y)
            total_loss += float(loss.item())
            total_iou += float(iou_score(logits, y).item())
            total_f1 += float(f1.item())
            total_precision += float(precision.item())
            total_recall += float(recall.item())
    n = max(len(loader), 1)
    return {
        "loss": total_loss / n,
        "iou": total_iou / n,
        "f1": total_f1 / n,
        "precision": total_precision / n,
        "recall": total_recall / n,
    }


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
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError("Empty train/val split. Build more samples or adjust holdout regimes.")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["detector"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["detector"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )

    model = build_cloud_detector(
        in_channels=len(cfg["data"]["sentinel2_bands"]),
        encoder=cfg["training"]["detector"]["encoder"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["detector"]["lr"]), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["detector"]["epochs"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_iou = -1.0
    history = []

    for epoch in range(1, int(cfg["training"]["detector"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, True)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, False)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(row)
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), Path(cfg["paths"]["model_dir"]) / "best_cloud_detector.pt")
            print(f"Saved best detector IoU={best_iou:.4f}")

    save_json(history, Path(cfg["paths"]["result_dir"]) / "detector_history.json")


if __name__ == "__main__":
    main()


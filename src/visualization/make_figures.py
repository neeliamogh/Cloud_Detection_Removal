from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.patch_dataset import CloudPatchDataset
from src.models.remover import MultiModalUNet
from src.utils import load_config, read_json


def rgb(x: np.ndarray) -> np.ndarray:
    # Sentinel-2 order in config: B02, B03, B04, B08, B11, B12.
    img = np.stack([x[2], x[1], x[0]], axis=-1)
    return np.clip(img, 0, 1)


def plot_history(history_file, title_prefix, out_path):
    if not history_file.exists():
        return
    history = read_json(history_file)
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train"]["loss"] for r in history]
    val_loss = [r["val"]["loss"] for r in history]
    
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, train_loss, label="Train Loss", color="blue")
    ax1.plot(epochs, val_loss, label="Val Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    
    ax2 = ax1.twinx()
    if "iou" in history[0]["val"]:
        val_metric = [r["val"]["iou"] for r in history]
        metric_name = "IoU"
    else:
        val_metric = [r["val"]["psnr"] for r in history]
        metric_name = "PSNR"
        
    ax2.plot(epochs, val_metric, label=f"Val {metric_name}", color="green")
    ax2.set_ylabel(metric_name)
    ax2.legend(loc="upper right")
    
    plt.title(f"{title_prefix} Training Curve")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result_dir = Path(cfg["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Training curves
    plot_history(result_dir / "detector_history.json", "Detector", result_dir / "detector_curve.png")
    plot_history(result_dir / "remover_history.json", "Remover", result_dir / "remover_curve.png")

    eval_path = result_dir / "evaluation_summary.json"
    if eval_path.exists():
        rows = read_json(eval_path)
        test_row = next((r for r in rows if r["split"] == "test"), None)
        if test_row:
            # Per-cloud-fraction metrics
            buckets = ["low", "medium", "high", "very_high"]
            psnrs = [test_row.get(f"{b}_psnr", 0) for b in buckets]
            maes = [test_row.get(f"{b}_mae", 0) for b in buckets]
            
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(buckets, psnrs, marker="o", color="blue", label="PSNR")
            ax1.set_ylabel("PSNR")
            ax2 = ax1.twinx()
            ax2.plot(buckets, maes, marker="s", color="red", label="MAE")
            ax2.set_ylabel("MAE")
            plt.title("Metrics by Cloud Fraction Bucket (Test Split)")
            fig.tight_layout()
            fig.savefig(result_dir / "cloud_fraction_metrics.png", dpi=180)
            plt.close(fig)

            # Per-regime metrics
            regimes = []
            regime_psnrs = []
            for k in test_row.keys():
                if k.startswith("regime_") and k.endswith("_psnr"):
                    regime = k[7:-5]
                    regimes.append(regime)
                    regime_psnrs.append(test_row[k])
            
            if regimes:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(regimes, regime_psnrs, color="#2f6f8f")
                ax.set_ylabel("PSNR")
                ax.set_title("Per-Regime PSNR (Test Split)")
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(result_dir / "regime_metrics.png", dpi=180)
                plt.close(fig)

    stats = Path(cfg["paths"]["data_root"]) / "normalization_stats.json"
    split = Path(cfg["paths"]["split_dir"]) / "test.json"
    items = read_json(split)
    if not items:
        print("No test items for figures.")
        return
    ds = CloudPatchDataset(split, stats)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalUNet(
        in_channels=len(cfg["data"]["sentinel2_bands"]) + len(cfg["data"]["sentinel1_bands"]) + 1,
        out_channels=len(cfg["data"]["sentinel2_bands"]),
    ).to(device)
    
    model_path = Path(cfg["paths"]["model_dir"]) / "best_cloud_remover.pt"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Qualitative grid (3 per regime)
    regime_samples = {}
    for i in range(len(ds)):
        r = ds[i]["regime"]
        if r not in regime_samples:
            regime_samples[r] = []
        if len(regime_samples[r]) < 3:
            regime_samples[r].append(i)

    selected_indices = [idx for indices in regime_samples.values() for idx in indices]
    n = len(selected_indices)
    if n == 0:
        return
        
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[None, :]
        
    for i, idx in enumerate(selected_indices):
        sample = ds[idx]
        pred = model(sample["remover_x"][None].to(device)).cpu().numpy()[0]
        cloudy = sample["detector_x"].numpy()
        target = sample["target"].numpy()
        mask = sample["mask"].numpy()[0]
        
        regime = sample["regime"]
        
        axes[i, 0].imshow(rgb(cloudy))
        axes[i, 0].set_title(f"Cloudy S2 ({regime})")
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Cloud mask")
        axes[i, 2].imshow(rgb(pred))
        axes[i, 2].set_title("Reconstructed")
        axes[i, 3].imshow(rgb(target))
        axes[i, 3].set_title("Clear target")
        for ax in axes[i]:
            ax.axis("off")
            
    fig.tight_layout()
    fig.savefig(result_dir / "qualitative_results.png", dpi=180)
    plt.close(fig)
    print("Figures saved.")

if __name__ == "__main__":
    main()

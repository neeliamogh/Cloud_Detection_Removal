from __future__ import annotations

import torch
import torch.nn.functional as F


def iou_score(logits, target, threshold: float = 0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + 1e-6) / (union + 1e-6)).mean()


def f1_score(logits, target, threshold: float = 0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    target = (target > 0.5).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1, precision, recall


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def mae(pred, target):
    return torch.abs(pred - target).mean()


def rmse(pred, target):
    return torch.sqrt(F.mse_loss(pred, target) + 1e-8)


def sam(pred, target):
    p = pred.flatten(2)
    t = target.flatten(2)
    cos = (p * t).sum(1) / (torch.linalg.norm(p, dim=1) * torch.linalg.norm(t, dim=1) + 1e-8)
    return torch.acos(torch.clamp(cos, -0.999999, 0.999999)).mean()


def ndvi_error(pred, target, red_idx: int = 2, nir_idx: int = 3):
    pred_ndvi = (pred[:, nir_idx] - pred[:, red_idx]) / (pred[:, nir_idx] + pred[:, red_idx] + 1e-6)
    target_ndvi = (target[:, nir_idx] - target[:, red_idx]) / (target[:, nir_idx] + target[:, red_idx] + 1e-6)
    return torch.abs(pred_ndvi - target_ndvi).mean()


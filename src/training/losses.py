from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)
        sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return 1.0 - ssim.mean()


def spectral_angle_loss(pred, target):
    pred_v = pred.flatten(2)
    target_v = target.flatten(2)
    dot = (pred_v * target_v).sum(dim=1)
    pred_norm = torch.linalg.norm(pred_v, dim=1)
    target_norm = torch.linalg.norm(target_v, dim=1)
    cos = dot / (pred_norm * target_norm + 1e-8)
    return torch.acos(torch.clamp(cos, -0.999999, 0.999999)).mean()


def cloud_weighted_l1(pred, target, mask, cloud_weight: float):
    weights = 1.0 + cloud_weight * mask
    return (torch.abs(pred - target) * weights).mean()


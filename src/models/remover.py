from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn


class MultiModalUNet(nn.Module):
    """Reconstruction model for S2 + S1 + mask using SMP."""

    def __init__(self, in_channels: int = 9, out_channels: int = 6, base: int = 64):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation="sigmoid",
        )

    def forward(self, x):
        return self.model(x)

from __future__ import annotations

import segmentation_models_pytorch as smp


def build_cloud_detector(in_channels: int = 6, encoder: str = "resnet34", encoder_weights: str | None = "imagenet"):
    try:
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )
    except Exception as exc:
        if encoder_weights is None:
            raise
        print(f"WARNING: Could not load pretrained encoder weights ({exc}). Falling back to random init.")
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

# Baselines And Ablations

For a serious paper, add these experiments after the main run works:

## Baselines

- Previous 38-Cloud U-Net detector.
- pix2pix RGB cloud removal.
- Optical-only U-Net reconstruction.
- SAR-only reconstruction.
- Cloud-mask-only optical reconstruction.
- Multi-modal Sentinel-1/Sentinel-2 reconstruction.

## Ablations

- Without Sentinel-1.
- Without cloud-region weighted loss.
- Without SAM loss.
- Without NDVI evaluation.
- Random split versus cross-regime split.
- Short temporal gap versus long temporal gap.

## Reviewer-Relevant Claims

Acceptable:

- The method improves real-pair reconstruction over Indian cross-regime holdouts.
- Sentinel-1 guidance improves cloud-region reflectance reconstruction.
- Cross-regime validation reveals stronger failure modes than random splits.

Not acceptable:

- The method recovers true ground perfectly under all thick clouds.
- The method is real-time.
- Visual RGB quality alone proves scientific correctness.


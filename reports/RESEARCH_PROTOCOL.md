# Research Protocol

## Title Candidate

Real Non-Synthetic Cloud Detection and Cloud-Contaminated Reflectance Reconstruction over Indian Terrain Using Sentinel-1/Sentinel-2 Cross-Regime Validation

## Why This Is Stronger Than The Previous Project

The previous 38-Cloud + pix2pix setup was a valid prototype, but it had four research weaknesses:

1. Landsat-only segmentation data.
2. RGB-only reconstruction.
3. Random split evaluation.
4. Weak cloud-removal realism for thick clouds.

This pipeline shifts the work to real paired Sentinel observations over Indian regimes, with Sentinel-1 SAR as an auxiliary signal and cross-regime validation.

## Dataset Construction

The main dataset is created on Lightning AI from Microsoft Planetary Computer STAC.

Each sample contains:

- Cloudy Sentinel-2 L2A surface reflectance.
- Clear Sentinel-2 L2A reference within a configurable temporal window.
- Sentinel-1 RTC VV/VH close to the cloudy acquisition.
- Cloud/shadow mask from Sentinel-2 SCL.
- Indian terrain regime metadata.

No synthetic cloudy images are generated.

## Indian Cross-Regime Design

Regimes:

- Himalayas
- Indo-Gangetic Plain
- Western Ghats
- Northeast India
- Deccan Plateau
- Thar Desert
- Coastal India
- Urban NCR

Default holdout regimes:

- Northeast India
- Thar Desert

This tests whether the model generalizes to persistent cloud/tropical terrain and bright arid terrain.

## Model

Cloud detector:

- U-Net with ImageNet encoder from segmentation-models-pytorch.
- Input: multispectral Sentinel-2 bands.
- Output: cloud/shadow mask.

Cloud remover:

- Multi-modal U-Net.
- Input: cloudy Sentinel-2 bands, Sentinel-1 VV/VH, cloud mask.
- Output: reconstructed Sentinel-2 reflectance bands.

## Metrics

Detection:

- IoU
- F1
- Precision
- Recall

Reconstruction:

- PSNR
- MAE
- RMSE
- SAM
- NDVI error
- Cloud-region MAE

## Q1-Level Expectations

To approach Q1-level quality, the final manuscript should include:

- A large enough India dataset.
- Cross-regime validation.
- External validation.
- Baseline comparison against pix2pix, optical-only U-Net, and SAR-guided model.
- Ablation study.
- Downstream vegetation-index preservation.
- Clear limitations around thick-cloud ambiguity.

### Expected Metric Targets

Based on the latest literature in cloud removal and detection, we are targeting:
- **Detector**: IoU > 0.65, F1 > 0.75
- **Remover**: PSNR > 22 dB, MAE < 0.08, SAM < 0.15, NDVI error < 0.10

### Statistical Reporting

For all metric reporting, we will report the mean $\pm$ standard deviation across all patches within each evaluation regime. This provides a clear confidence interval and statistical rigor, ensuring that results are robust across both bright arid (e.g., Thar Desert) and persistent-cloud tropical (e.g., Northeast India) regimes.

This code package provides a stable research-run foundation. The paper quality depends on scaling final dataset size, adding more baselines, and reporting statistical confidence after the pipeline is stable.

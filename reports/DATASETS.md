# Dataset Plan

## Main Dataset

Custom Indian Sentinel-1/Sentinel-2 paired dataset built directly on Lightning AI using STAC access.

Default source:

- Microsoft Planetary Computer
- Sentinel-2 L2A
- Sentinel-1 RTC

## Recommended External Benchmarks

Use these for paper-grade validation when credentials/storage are available:

- CloudSEN12+ for cloud/shadow detection.
- SEN12MS-CR for SAR-optical cloud removal.
- AllClear for large-scale cloud-gap reconstruction benchmarking.
- SPARCS / Landsat CCA for external cloud-mask validation.

Some datasets require account-specific download steps. This repository does not hide that behind fake code. Use the official dataset licenses and credentials, then convert them into the same `.npz` schema used by `CloudPatchDataset`.

## NPZ Schema

Each patch file should contain:

```text
cloudy_s2: float32, shape [C_s2, H, W]
clear_s2: float32, shape [C_s2, H, W]
s1: float32, shape [2, H, W]
cloud_mask: float32, shape [1, H, W]
regime: string
cloudy_date: string
clear_date: string
temporal_gap_days: int
cloud_fraction: float
```


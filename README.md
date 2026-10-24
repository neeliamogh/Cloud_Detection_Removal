# Cloud Removal India Research Pipeline

Lightning-ready deep learning pipeline for real, non-synthetic cloud detection and cloud-contaminated reflectance reconstruction over Indian terrain.

This repository is designed to be uploaded to Lightning AI Studio. Dataset downloads and processing happen on the Lightning server, not on your local desktop.

## Research Position

This project targets a stronger research formulation than RGB pix2pix cloud removal:

- Real Sentinel-2 / Sentinel-1 / HLS data only.
- Indian terrain-specific dataset construction.
- Cross-regime validation across Indian agro-climatic and terrain regimes.
- External validation using public benchmarks where accessible.
- Scientific reconstruction metrics, not only visual RGB quality.

## Main Pipeline

1. Download/build real Indian Sentinel-1 and Sentinel-2 paired samples.
2. Preprocess, co-register, normalize, and tile patches.
3. Train cloud detector.
4. Train cloud-removal/reconstruction model.
5. Evaluate using internal, cross-regime, and external validation.
6. Generate paper-ready metrics, figures, logs, and result zip.

## Lightning Quick Start

```bash
unzip cloud_removal_india.zip
cd cloud_removal_india
chmod +x scripts/*.sh
bash scripts/setup_lightning.sh
nohup bash scripts/run_full_pipeline.sh > logs/full_pipeline.log 2>&1 &
tail -f logs/full_pipeline.log
```

## Recommended First Check

Before spending the full GPU block, run a tiny end-to-end check:

```bash
nohup bash scripts/run_fast_check.sh > logs/fast_check.log 2>&1 &
tail -f logs/fast_check.log
```

If this finishes, run the full pipeline.

## Datasets Used By The Code

The executable pipeline directly uses:

- Sentinel-2 L2A surface reflectance from Microsoft Planetary Computer.
- Sentinel-1 RTC VV/VH SAR from Microsoft Planetary Computer.
- Sentinel-2 SCL cloud/shadow labels for cloud-mask supervision and filtering.

The research protocol also recommends external validation with CloudSEN12+, SEN12MS-CR, AllClear, and SPARCS/Landsat CCA. Those are documented as external benchmarks because some require separate credentials, license handling, or large storage.

## Result Download Command

After training finishes:

```bash
bash scripts/package_results.sh
```

Download:

```text
/teamspace/studios/this_studio/cloud_removal_india/results/cloud_removal_india_results.zip
```

## Four-Hour L40S Profile

The default config is tuned for a complete download-to-results run inside roughly 4 hours on one L40S:

- 160x160 real patches.
- 4 Sentinel-2 bands: blue, green, red, NIR.
- Sentinel-1 VV/VH auxiliary input.
- 25 target samples per Indian regime.
- Bounded STAC scene scans.
- 3 detector epochs.
- 5 remover epochs.

This is a strong prototype/research run, not a final foundation-scale training run. For a paper submission, scale sample counts and epochs after this pipeline is stable.

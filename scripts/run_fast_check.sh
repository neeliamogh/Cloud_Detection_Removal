#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/cloud_removal_india
mkdir -p logs results_fast models_fast data_fast/raw data_fast/patches data_fast/splits

echo "Started fast check at $(date)"
nvidia-smi || true

echo "[FAST CHECK 0/7] Diagnosing Sentinel-2 raster access"
python -m src.data.diagnose_stac_access --config configs/fast_check.yaml || true

echo "[FAST CHECK 1/7] Building a tiny real Indian Sentinel-1/Sentinel-2 patch dataset"
python -m src.data.build_india_stac_dataset --config configs/fast_check.yaml

echo "[FAST CHECK 2/7] Computing normalization statistics"
python -m src.preprocessing.compute_stats --config configs/fast_check.yaml

echo "[FAST CHECK 3/7] Creating splits"
python -m src.data.make_splits --config configs/fast_check.yaml

echo "[FAST CHECK 4/7] Training detector for one epoch"
python -m src.training.train_detector --config configs/fast_check.yaml

echo "[FAST CHECK 5/7] Training remover for one epoch"
python -m src.training.train_remover --config configs/fast_check.yaml

echo "[FAST CHECK 6/7] Evaluating"
python -m src.evaluation.evaluate_all --config configs/fast_check.yaml

echo "[FAST CHECK 7/7] Making figures"
python -m src.visualization.make_figures --config configs/fast_check.yaml

echo "Fast check finished. If this succeeds, run scripts/run_full_pipeline.sh."

#!/usr/bin/env bash
set -euo pipefail

cd /teamspace/studios/this_studio/cloud_removal_india
mkdir -p logs results models data/raw data/patches data/splits reports

START_TS=$(date +%s)
echo "Started full 4-hour pipeline at $(date)"
echo "GPU status:"
nvidia-smi || true

echo "[1/7] Building real Indian Sentinel-1/Sentinel-2 patch dataset"
python -m src.data.build_india_stac_dataset --config configs/default.yaml

echo "[2/7] Computing robust normalization statistics"
python -m src.preprocessing.compute_stats --config configs/default.yaml

echo "[3/7] Creating cross-regime splits"
python -m src.data.make_splits --config configs/default.yaml

echo "[4/7] Training cloud detector"
python -m src.training.train_detector --config configs/default.yaml

echo "[5/7] Training cloud removal model"
python -m src.training.train_remover --config configs/default.yaml

echo "[5.5/7] Training and evaluating baselines"
python -m src.training.train_baselines --config configs/default.yaml

echo "[6/7] Running internal, cross-regime, and external-style evaluation"
python -m src.evaluation.evaluate_all --config configs/default.yaml

echo "[7/7] Generating figures and result package"
python -m src.visualization.make_figures --config configs/default.yaml
bash scripts/package_results.sh

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo "Pipeline finished at $(date). Elapsed seconds: ${ELAPSED}"

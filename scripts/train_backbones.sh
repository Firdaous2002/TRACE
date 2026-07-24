#!/usr/bin/env bash
# Train all 15 (model, dataset) backbones used in the paper.
# Checkpoints land in ./checkpoints/<model>_<dataset>.pt .

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON_BIN:-python}"

MODELS=(sasrec bert4rec gru4rec)
DATASETS=(ml-100k ml-1m steam beauty sports)

# Prerequisite: preprocessed datasets must exist under data/processed/
missing_data=()
for ds in "${DATASETS[@]}"; do
  if [[ ! -d "data/processed/$ds" ]]; then
    missing_data+=("$ds")
  fi
done
if (( ${#missing_data[@]} > 0 )); then
  echo "[error] Preprocessed datasets missing: ${missing_data[*]}" >&2
  echo "Run: make data   (or: python -m data.preprocess --dataset <name> for each)" >&2
  exit 1
fi

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    echo "[train] $model / $ds"
    "$PY" run_train.py model.name="$model" data.dataset="$ds"
  done
done

#!/usr/bin/env bash
# Reproduce the main bring-in and push-out experiments (Sections 5.2-5.3 of the paper)
# for TRACE, GECE, Greedy and Random on all 15 (model, dataset) combinations and
# seeds {42, 43, 44}. Results are written to ./results/ as
# explain_<model>_<dataset>_<explainer>_<target>_seed<seed>.json and
# aggregated via aggregate_summary.py.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON_BIN:-python}"

MODELS=(sasrec bert4rec gru4rec)
DATASETS=(ml-100k ml-1m steam beauty sports)
SEEDS=(42 43 44)

# Prerequisite: all 15 backbone checkpoints must exist
missing_ckpt=()
for m in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    if [[ ! -f "checkpoints/${m}_${ds}.pt" ]]; then
      missing_ckpt+=("${m}_${ds}.pt")
    fi
  done
done
if (( ${#missing_ckpt[@]} > 0 )); then
  echo "[error] Missing backbone checkpoints (${#missing_ckpt[@]}): ${missing_ckpt[*]}" >&2
  echo "Run: make train   (or: bash scripts/train_backbones.sh)" >&2
  exit 1
fi

N_USERS="${N_USERS:-500}"
Q_BUDGET="${Q_BUDGET:-10000}"
K_MAX="${K_MAX:-5}"

run_trace() {
  local model=$1 dataset=$2 constraint=$3 rank_min=$4 rank_max=$5 seed=$6
  "$PY" run_explain.py \
    model.name="$model" data.dataset="$dataset" \
    explainer.name=beam \
    targeted.enabled=true targeted.constraint="$constraint" \
    targeted.rank_min="$rank_min" targeted.rank_max="$rank_max" \
    beam.beam_width=5 beam.k_context=6 beam.k_target_nn=3 beam.k_local_nn=3 \
    beam.candidate_strategy=pacer_lite beam.position_strategy=influence \
    beam.position_top_m=5 beam.pacer_shortlist_size=12 \
    explainer.max_edits="$K_MAX" explainer.query_budget="$Q_BUDGET" \
    evaluation.n_users="$N_USERS" evaluation.n_seeds=1 seed="$seed"
}

run_baseline() {
  local model=$1 dataset=$2 explainer=$3 constraint=$4 rank_min=$5 rank_max=$6 seed=$7
  "$PY" run_explain.py \
    model.name="$model" data.dataset="$dataset" \
    explainer.name="$explainer" \
    targeted.enabled=true targeted.constraint="$constraint" \
    targeted.rank_min="$rank_min" targeted.rank_max="$rank_max" \
    explainer.max_edits="$K_MAX" explainer.query_budget="$Q_BUDGET" \
    evaluation.n_users="$N_USERS" evaluation.n_seeds=1 seed="$seed"
}

for model in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      # bring-in (target = uniform rank in [2,50], push to top-1)
      run_trace    "$model" "$ds" bring_in 2 50 "$seed"
      for m in ga greedy random; do
        run_baseline "$model" "$ds" "$m" bring_in 2 50 "$seed"
      done
      # push-out (current top-1 must fall below top-50)
      run_trace    "$model" "$ds" push_out 1 1 "$seed"
      for m in ga greedy random; do
        run_baseline "$model" "$ds" "$m" push_out 1 1 "$seed"
      done
    done
  done
done

echo "[aggregate] compiling per-combo summaries"
"$PY" aggregate_summary.py --results-dir ./results

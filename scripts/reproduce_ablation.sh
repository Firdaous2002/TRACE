#!/usr/bin/env bash
# Reproduce the TRACE cumulative ablation (Table 4 / Table 5) on the three
# representative combinations: SASRec/ML-100K, BERT4Rec/ML-1M, GRU4Rec/Steam.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PY="${PYTHON_BIN:-python}"

COMBOS=("sasrec ml-100k" "bert4rec ml-1m" "gru4rec steam")
SEEDS=(42 43 44)

# Prerequisite: the three ablation combo checkpoints
missing_ckpt=()
for combo in "${COMBOS[@]}"; do
  read -r _m _d <<<"$combo"
  if [[ ! -f "checkpoints/${_m}_${_d}.pt" ]]; then
    missing_ckpt+=("${_m}_${_d}.pt")
  fi
done
if (( ${#missing_ckpt[@]} > 0 )); then
  echo "[error] Missing ablation checkpoints: ${missing_ckpt[*]}" >&2
  echo "Run: make train   (or train the three combos individually)" >&2
  exit 1
fi

run_variant() {
  local model=$1 dataset=$2 seed=$3 variant=$4
  local beam_width pos_strategy cand_strategy shortlist k_context k_target k_local
  case "$variant" in
    naive)       beam_width=1; pos_strategy=all;       cand_strategy=pool;        shortlist=0;  k_context=6; k_target=0; k_local=0 ;;
    beam)        beam_width=5; pos_strategy=all;       cand_strategy=pool;        shortlist=0;  k_context=6; k_target=0; k_local=0 ;;
    influence)   beam_width=5; pos_strategy=influence; cand_strategy=pool;        shortlist=0;  k_context=6; k_target=0; k_local=0 ;;
    full)        beam_width=5; pos_strategy=influence; cand_strategy=pacer_lite;  shortlist=12; k_context=6; k_target=3; k_local=3 ;;
    *) echo "unknown variant $variant" >&2; exit 1 ;;
  esac

  for task in bringin pushout; do
    local constraint rank_min rank_max
    if [[ $task == bringin ]]; then constraint=bring_in; rank_min=2; rank_max=50
    else                             constraint=push_out; rank_min=1; rank_max=1
    fi

    "$PY" run_explain.py \
      model.name="$model" data.dataset="$dataset" \
      explainer.name=beam \
      targeted.enabled=true targeted.constraint="$constraint" \
      targeted.rank_min="$rank_min" targeted.rank_max="$rank_max" \
      beam.beam_width="$beam_width" \
      beam.k_context="$k_context" beam.k_target_nn="$k_target" beam.k_local_nn="$k_local" \
      beam.candidate_strategy="$cand_strategy" beam.position_strategy="$pos_strategy" \
      beam.position_top_m=5 beam.pacer_shortlist_size="$shortlist" \
      explainer.max_edits=5 evaluation.n_users=500 evaluation.n_seeds=1 \
      seed="$seed" evaluation.run_tag="ablation-${variant}"
  done
}

for combo in "${COMBOS[@]}"; do
  read -r model dataset <<<"$combo"
  for seed in "${SEEDS[@]}"; do
    for variant in naive beam influence full; do
      run_variant "$model" "$dataset" "$seed" "$variant"
    done
  done
done
